%h5disp('vgg19.h5');
%load('count.mat');
train_set=h5read('vgg19.h5','/test_set');
%f2=h5read('vgg19.h5','/validation_set');
%train_set=cat(4,f1,f2);
[t1,t2,t3,t4]=size(train_set);
train_new=single(zeros(t4,t2*t3,t1));
for i=1:t4
    temp1=reshape(train_set(:,:,:,i),512,49);
    train_new(i,:,:)=temp1';
end
h5create('vgg19_bolck5_new.h5','/test',size(train_new));
h5write('vgg19_bolck5_new.h5','/test',train_new);
h5disp('vgg19_bolck5_new.h5');