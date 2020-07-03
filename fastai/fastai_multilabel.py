from net.resnet50_cam import CamNet
from fastai.vision import *
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
        data_dir = "D:\\workspace\\dataset\\adas_parsing_data_0409_to0416\\voc_fmt"
        dl_tfms = get_transforms(max_lighting=0.1, max_zoom=1.05, max_warp=0.)

        data = (ImageList.from_csv(data_dir, 'label.csv', folder='JPEGImages', suffix='.jpg')
                .split_by_rand_pct()
                .label_from_df(label_delim=' ')
                .transform(dl_tfms, size=512)
                .databunch(bs=8) # todo change to 8
                .normalize(imagenet_stats))

        net = CamNet()
        learn = Learner(data, net, metrics=MultiLabelFbeta(beta=2))
        learn.fit(5)
