import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchsummary import summary
from utils.utils import build_targets, to_cpu, non_max_suppression

class convolu3(torch.nn.Module):
    def __init__(self, Kn_1, Kn_2):
        super(convolu3, self).__init__()
        self.conv3=torch.nn.Conv2d(Kn_1, Kn_2, 3,padding=1)
        self.bn3=torch.nn.BatchNorm2d(Kn_2,momentum=0.9)
        self.relu3=torch.nn.LeakyReLU(0.1)
    def forward(self,x):
        y=self.conv3(x)
        y=self.bn3(y)
        y=self.relu3(y)
        return y

class convolu3down(torch.nn.Module):
    def __init__(self, Kn_1, Kn_2):
        super(convolu3down, self).__init__()
        self.conv3=torch.nn.Conv2d(Kn_1, Kn_2, 3,padding=1,stride=2)
        self.bn3=torch.nn.BatchNorm2d(Kn_2,momentum=0.9)
        self.relu3=torch.nn.LeakyReLU(0.1)
    def forward(self,x):
        y=self.conv3(x)
        y=self.bn3(y)
        y=self.relu3(y)
        return y

class convolu1(torch.nn.Module):
    def __init__(self, Dn,Kn_1):
        super(convolu1, self).__init__()
        self.conv1=torch.nn.Conv2d(Dn, Kn_1, 1)
        self.bn1=torch.nn.BatchNorm2d(Kn_1,momentum=0.9)
        self.relu1=torch.nn.LeakyReLU(0.1)
    def forward(self,x):
        y=self.conv1(x)
        y=self.bn1(y)
        y=self.relu1(y)
        return y

class dark_res(torch.nn.Module):
    def __init__(self, Dn,Kn_1, Kn_2):
        super(dark_res, self).__init__()
        self.conv1=convolu1(Dn, Kn_1)
        self.conv3=convolu3(Kn_1, Kn_2)
        self.ln3=torch.nn.Linear(Kn_1, Kn_2)
    def forward(self,x):
        residual = x
        y=self.conv1(x)
        y=self.conv3(y)
        y+=residual
        return y

class dark_link(torch.nn.Module):
    def __init__(self, Dn,Kn_1, Kn_2,nub):
        super(dark_link, self).__init__()
        self.darkres=[dark_res(Dn,Kn_1, Kn_2).cuda() for i in range(nub)]
    def forward(self,x):
        y=x
        for darkresnet in  self.darkres:
            y=darkresnet(y)
        return y

class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=416):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss( [obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss

class dark_net(torch.nn.Module):
    def __init__(self,class_nub):
        super(dark_net, self).__init__()
        self.convs1=convolu3(3,32)
        self.convd1=convolu3down(32,64)
        self.dark_link1=dark_link(64,32,64,1)
        self.convd2=convolu3down(64,128)
        self.dark_link2=dark_link(128,64,128,2)
        self.convd3=convolu3down(128,256)
        self.dark_link3=dark_link(256,128,256,8)
        self.convd4=convolu3down(256,512)
        self.dark_link4=dark_link(512,256,512,8)
        self.convd5=convolu3down(512,1024)
        self.dark_link5=dark_link(1024,512,1024,4)


        self.od11=dark_link(1024,512,1024,3)
        self.od12=dark_link(1024,512,1024,1)
        self.o1=convolu1(1024,3*(5+class_nub))
        self.yolo1=YOLOLayer([(116,90),  (156,198),  (373,326)],80)

        self.oo2=convolu1(1024,256)
        self.up2=Upsample(scale_factor=2, mode="nearest")
        self.od21=convolu3(768,512)
        self.od22=dark_link(512,256,512,2)
        self.od23=dark_link(512,256,512,1)
        self.o2=convolu1(512,3*(5+class_nub))
        self.yolo2=YOLOLayer([(30,61),  (62,45),  (59,119)],80)

        self.oo3=convolu1(512,128)
        self.up3=Upsample(scale_factor=2, mode="nearest")
        self.od31=convolu3(384,256)
        self.od32=dark_link(256,128,256,2)
        self.od33=dark_link(256,128,256,1)
        self.o3=convolu1(256,3*(5+class_nub))
        self.yolo3=YOLOLayer([(10,13),  (16,30),  (33,23)],80)

    def forward(self,x):
        y1=self.convs1(x)
        y2=self.convd1(y1)
        y3=self.dark_link1(y2)
        y4=self.convd2(y3)
        y5=self.dark_link2(y4)
        y6=self.convd3(y5)
        y7=self.dark_link3(y6)
        y8=self.convd4(y7)
        y9=self.dark_link4(y8)
        y10=self.convd5(y9)
        y11=self.dark_link5(y10)

        o11=self.od11(y11)
        o12=self.od12(o11)
        out1=self.o1(o12)
        yolo1o,loss1=self.yolo1(out1,targets)

        oo21=self.oo2(o11)
        ups2=self.up2(oo21)
        round2=torch.cat([y9, ups2], 1)
        odo21=self.od21(round2)
        odo22=self.od22(odo21)
        odo23=self.od23(odo22)
        out2=self.o2(odo23)
        yolo2o,loss2=self.yolo2(out2,targets)

        oo31=self.oo3(odo22)
        ups3=self.up3(oo31)
        round3=torch.cat([y7, ups3], 1)
        odo31=self.od31(round3)
        odo32=self.od32(odo31)
        odo33=self.od33(odo32)
        out3=self.o3(odo33)
        yolo3o,loss3=self.yolo3(out3,targets)
        return [yolo1o,yolo2o,yolo3o],loss1+loss2+loss3



'''net=dark_net(80).cuda()
#net=dark_res(3,128,128).cuda()
#summary(net, (3,416,416))
input_arr=torch.rand(1,3,416,416).cuda()
with SummaryWriter(comment='net') as w:
    w.add_graph(net,input_arr)'''
