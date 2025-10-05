import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.layers import ResLayer
from mmyolo.registry import MODELS
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 dilation=1, bias=False, conv_cfg=None, norm_cfg=dict(type='BN')):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积 - 逐通道卷积
        self.depthwise_conv = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)
        
        # 点卷积 - 1x1卷积融合特征
        self.pointwise_conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)
        
        # 批归一化层
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_channels, postfix=2)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)
        
        self.relu = nn.ReLU(inplace=True)
        
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    @property
    def norm2(self):
        return getattr(self, self.norm2_name)
    
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.pointwise_conv(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        return out

class MobileBasicBlock(BaseModule):
    """使用深度可分离卷积的基础残差块"""
    expansion = 1
    
    def __init__(self, 
                 inplanes, 
                 planes, 
                 stride=1, 
                 dilation=1, 
                 downsample=None, 
                 style='pytorch', 
                 with_cp=False, 
                 conv_cfg=None, 
                 norm_cfg=dict(type='BN'), 
                 dcn=None, 
                 plugins=None, 
                 init_cfg=None):
        super(MobileBasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        
        # 使用深度可分离卷积替换普通3x3卷积
        self.conv1 = DepthwiseSeparableConv(
            inplanes, 
            planes, 
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        
        # 第二个卷积层也是深度可分离卷积
        self.conv2 = DepthwiseSeparableConv(
            planes, 
            planes, 
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
    
    def forward(self, x):
        """Forward function."""
        
        def _inner_forward(x):
            identity = x
            
            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            
            out = out + identity
            return out
        
        if self.with_cp and x.requires_grad:
            out = torch.utils.checkpoint.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        
        out = self.relu(out)
        return out

@MODELS.register_module()
class MobileResNet(BaseModule):
    """使用深度可分离卷积的ResNet主干网络"""
    
    arch_settings = {
        18: (MobileBasicBlock, (2, 2, 2, 2)),
        34: (MobileBasicBlock, (3, 4, 6, 3))
    }
    
    def __init__(self, 
                 depth=18, 
                 in_channels=3, 
                 stem_channels=None, 
                 base_channels=64, 
                 num_stages=4, 
                 strides=(1, 2, 2, 2), 
                 dilations=(1, 1, 1, 1), 
                 out_indices=(0, 1, 2, 3), 
                 style='pytorch', 
                 deep_stem=False, 
                 avg_down=False, 
                 frozen_stages=-1, 
                 conv_cfg=None, 
                 norm_cfg=dict(type='BN', requires_grad=True), 
                 norm_eval=True, 
                 dcn=None, 
                 stage_with_dcn=(False, False, False, False), 
                 plugins=None, 
                 with_cp=False, 
                 zero_init_residual=True, 
                 pretrained=None, 
                 init_cfg=None):
        super(MobileResNet, self).__init__(init_cfg)
        
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for MobileResNet')
        
        self.depth = depth
        self.in_channels = in_channels
        
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        
        # 构建主干网络的stem层
        self._make_stem_layer(in_channels, stem_channels)
        
        # 构建residual layers
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins)
            
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
        
        self._freeze_stages()
        
        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)
    
    def make_stage_plugins(self, plugins, stage_idx):
        """创建stage级别的插件"""
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # 是否在当前stage插入插件
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)
        
        return stage_plugins
    
    def make_res_layer(self, **kwargs):
        """将stage中的所有block打包成一个ResLayer"""
        return ResLayer(**kwargs)
    
    def _make_stem_layer(self, in_channels, stem_channels):
        """构建网络的stem层"""
        if self.deep_stem:
            # 使用3个3x3卷积代替7x7卷积
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            # 使用标准的7x7卷积作为stem
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    @property
    def norm1(self):
        """返回名为'norm1'的归一化层"""
        return getattr(self, self.norm1_name)
    
    def _freeze_stages(self):
        """冻结指定的stage"""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.conv1.eval()
                self.norm1.eval()
                for param in self.conv1.parameters():
                    param.requires_grad = False
                for param in self.norm1.parameters():
                    param.requires_grad = False
        
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """前向传播函数"""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)


@MODELS.register_module()
class MobileResNet18(MobileResNet):
    """使用深度可分离卷积的ResNet-18模型，支持适配YOLOv8 neck输入

    """

    def __init__(self, **kwargs):
        # 固定深度为18，但暂时不设置base_channels
        super(MobileResNet18, self).__init__(18, base_channels=128, **kwargs)

        # 重置inplanes
        self.inplanes = self.stem_channels

        # 移除原有的res_layers
        for layer_name in self.res_layers:
            if hasattr(self, layer_name):
                delattr(self, layer_name)

        # 自定义各阶段的通道数
        custom_channels = [128, 256, 512, 512]

        # 重建res_layers
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None

            if self.plugins is not None:
                stage_plugins = self.make_stage_plugins(self.plugins, i)
            else:
                stage_plugins = None

            # 使用自定义通道数
            planes = custom_channels[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                plugins=stage_plugins)

            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # 更新feat_dim
        self.feat_dim = self.block.expansion * custom_channels[-1]

        # 设置输出索引（可选，根据需要调整）
        self.out_indices = (1, 2, 3)  # 输出所有四个阶段的特征
