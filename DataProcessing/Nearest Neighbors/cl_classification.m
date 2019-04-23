function [classes_sample_idx, classes] = cl_classification(classes_library, pos_x, pos_y)
% From a class library, group all samples to corresponding class(es).
% @INPUT: After the removal of reciprocal.
% class library. e.g., S1-library.mat : all user position with their RAs
% pos_x, pos_y. e.g., S1-pos.mat. : 1st column is index to pos before the removal of reciprocal, 2nd colum is the x or y - coordinates.
% @OUTPUT:
% classes_sample_idx. A cell, contains all classes and their members.
% classes. 4 columns: Encoded Classes, Tx, Rx, MCs 
% NOTE: classes_sample_idx and classes have the same order.
    classes = [];
    classes_sample_idx = {};
    modulus = size(classes_library,1);
    
    for i = 1:numel(classes_library);
        temp1 = classes_library(i);

        if (temp1 ~= 0) && ((isempty(classes)) ||  ... 
            isempty(find(classes(:,1)==temp1,1))) 

            bin_temp1 = de2bi(temp1,15);
            decoded_temp1 = [bi2de(bin_temp1(10:15)) bi2de(bin_temp1(5:9)) bi2de(bin_temp1(1:4))];
            classes = [classes; temp1 decoded_temp1];         %#ok<AGROW>
            temp2 = find(classes_library==classes_library(i));
            temp2_fix = (mod(temp2, modulus) + modulus*(mod(temp2, modulus)==0));
            sample_idx_temp = [temp2_fix, pos_x(temp2_fix,2), pos_y(temp2_fix,2)];
            classes_sample_idx = [classes_sample_idx; sample_idx_temp];    %#ok<AGROW>
        end
    end