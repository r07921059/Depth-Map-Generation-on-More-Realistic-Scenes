
if [ ! -f "classifier.pth" ]
then
    echo "File classifier.pth does not exist"
    wget -c https://www.dropbox.com/s/4799zh5wnbbl6jc/classifier.pth?dl=1 -O classifier.pth
else
    echo "File classifier.pth does is exist"
fi


if [ ! -f "pretrained_model_KITTI2012.tar" ]
then
    echo "File pretrained_model_KITTI2012.tar does not exist"
    wget -c https://www.dropbox.com/s/0utv5nhia7mmch0/pretrained_model_KITTI2012.tar?dl=1 -O pretrained_model_KITTI2012.tar
else
    echo "File pretrained_model_KITTI2012.tar does is exist"
fi

