%h5disp('vgg19.h5');
%load('count.mat');

train_set=h5read('vgg19.h5','/validation_set');
[t1,t2,t3,t4]=size(train_set);
train_new=single(zeros(t4,t2*t3,t1));
for i=1:t4
    temp1=reshape(train_set(:,:,:,i),512,49);
    temp1=temp1';
    train_new(i,:,:)=temp1;
end
h5create('vgg19_new.h5','/vali',size(train_new));
h5write('vgg19_new.h5','/vali',train_new);
h5disp('vgg19_new.h5');