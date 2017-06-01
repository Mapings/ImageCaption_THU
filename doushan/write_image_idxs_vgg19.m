load('train_vali_count.mat');
count=train_vali_count;
cum=cumsum(count);
image_idxs=zeros(sum(count),1);
for i=1:length(count)
    if i==1
        image_idxs(1:cum(i),:)=i;
    else
        image_idxs(cum(i-1)+1:cum(i),:)=i;
    end
end

h5create('vgg19_bolck5_new.h5','/image_idxs',size(image_idxs));
h5write('vgg19_bolck5_new.h5','/image_idxs',image_idxs);
h5disp('vgg19_bolck5_new.h5');
