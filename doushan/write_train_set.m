%h5disp('fc1.h5');
train_set=h5read('fc1.h5','/train_set');
train_set=train_set';
n=sum(count);
train_new=[];
for i=1:size(train_set,1)
    temp=repmat(train_set(i,:),count(i,:),1);
    train_new=[train_new;temp];
end

h5create('fc1_new.h5','/train_set',size(train_new));
hdf5write('fc1_new.h5','/train_set',train_new);
h5disp('fc1_new.h5');