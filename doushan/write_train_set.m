%h5disp('fc1.h5');
f1=h5read('fc1.h5','/train_set');
f2=h5read('fc1.h5','/validation_set');
load('train_vali_count.mat');
count1=train_vali_count;
train_set=cat(2,f1,f2)';
n=sum(count1);
cum_n=cumsum(count1);
train_new=single(zeros(n,size(train_set,2)));
for i=1:size(train_set,1)
    temp=repmat(train_set(i,:),count(i,:),1);
    if i==1
        train_new(1:cum_n(i),:)=temp;
    else
        train_new(cum_n(i-1)+1:cum_n(i),:)=temp;
    end
end

h5create('fc1_9000_new.h5','/train_set',size(train_new));
hdf5write('fc1_9000_new.h5','/train_set',train_new);
h5disp('fc1_9000_new.h5');