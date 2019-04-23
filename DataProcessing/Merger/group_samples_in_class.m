function number_classes_per_samples_for_considering_class = group_samples_in_class(class_i, number_classes_per_samples_for_all)  
% filter out samples in the coresponding class
% @INPUT: 
% class_i : considering class c_i
% number_classes_per_samples_all: a vector contain how many RAs to each samples. e.g., Sample1: 27000 28000 --> Sample1 belongs to 2 classes.
% @OUTPUT:
% number_classes_per_samples_for_considering_class: c_i

    iter_pos = size(class_i,1);
    number_classes_per_samples_for_considering_class = zeros(iter_pos,1);
    pos_idx = class_i(:,1);

    for j = 1:iter_pos
        number_classes_per_samples_for_considering_class(j) = number_classes_per_samples_for_all(pos_idx(j)); 
    end
