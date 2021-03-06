# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
from pysot.models.backbone.darts import darts
from pysot.models.backbone.darts_supernet import darts_supernet
from pysot.models.backbone.darts_latency_supernet import darts_latency_supernet

BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
              'darts': darts,
              'darts_supernet':darts_supernet,
              'darts_latency_supernet':darts_latency_supernet,
            }


def get_backbone(name, **kwargs):
    # print(kwargs)
    # return BACKBONES['resnet50'](**kwargs)
    return BACKBONES[name](**kwargs)
