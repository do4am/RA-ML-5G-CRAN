%load toy_classes.mat and working_pos.mat [toy_only]
% pos_x = pos_x(training_idx,:); %only training set
% pos_y = pos_y(training_idx,:); %only training set

pos = [pos_x(:,1:2), pos_y(:,2)]; %for all
pos_combined = pos; %for all

% pos_idx = find((pos(:,2)<=140.2) & (pos(:,2) >= 139) & (pos(:,3)<= 103)); %for toy
% pos_combined = pos(pos_idx,:); %for toy
% pos_x = pos_x(pos_idx,:); %for toy
% pos_y = pos_y(pos_idx,:); %for toy

ptCloud = pointCloud([pos_combined(:,1),pos_combined(:,2),zeros(length(pos_combined),1)]); %3-D
classes_lib = classes2_lib_updated;
                    %classes_library_updated(training_idx,:);
                    %toy_classes_lib;%classes2_lib_updated;%classes_library_updated;
                    %toy_classes_lib;%classes_library_updated;%toy_classes_lib_updated;

% %% only for training set
% [classes_sorted_idx2, classes_sorted2] = cl_classification(classes_lib, pos_x, pos_y);
% [classes_sorted2, classes_sorted_idx2] = cl_sort(classes_sorted2, classes_sorted_idx2);

%%
classes_idx = classes2_sorted_idx;
                          %toy_classes_idx;%classes_sorted_idx;%classes2_sorted_idx;
                          %toy_classes_idx;
                          
classes = classes2_sorted;%toy_classes;%classes_sorted;%classes2_sorted;%toy_classes;


%%
ClperV = zeros(size(classes_lib,1),2);
ClperV(:,1) = number_classes_per_sample(classes_lib);

n = length(classes_idx);
classes_size = zeros(n,2);
for i = 1:n
    classes_size(i,:) = [classes(i), length(classes_idx{i})];
end

%%
K = 3;  %No. of nearest neighbors
mode = 1; 
[classes_lib, classes_size_new] = cl_cluster(pos_combined, classes_lib, classes_size, ClperV, K, mode);

[classes_idx_new, classes_new] = cl_classification(classes_lib, pos_x, pos_y);
[classes_new, classes_idx_new] = cl_sort(classes_new, classes_idx_new);

%% Continue Reassignment
K = 3;
mode = 2;
ClperV = zeros(size(classes_lib,1),2);
ClperV(:,1) = number_classes_per_sample(classes_lib);
%repeatation
[classes_lib_new2, classes_size_new2] = cl_cluster(pos_combined, classes_lib, classes_size, ClperV, K, mode);

[classes_idx_new2, classes_new2] = cl_classification(classes_lib_new2, pos_x, pos_y);
[classes_new2, classes_idx_new2] = cl_sort(classes_new2, classes_idx_new2);

% % n = length(toy_classes_idx_new);
% % classes_size = zeros(n,2);
% % for i = 1:n
% %     classes_size(i,:) = [toy_classes_new(i), length(toy_classes_idx_new{i})];
% % end
% % sum(classes_size(:,2))

% %% training_only
% classes_lib_new2_training = classes_lib_new2;

%% remap classes_idx to compare with original 
n = size(classes,1);
toy_classes_idx_new_remap = cell(n,1);
for i = 1:n
    idx = find(classes_new2 == classes(i,1));
    if isempty(idx)
        toy_classes_idx_new_remap{i,1} = [];
    else
        toy_classes_idx_new_remap{i,1} = classes_idx_new2{idx,1};
    end    
end


%% Plotting Toy Clustered
    classes_data = toy_classes_idx_new_remap;%classes_sorted_idx2;%toy_classes_idx_new_remap;%classes2_sorted_idx;%toy_classes_idx_new_remap;%classes_sorted_idx; %toy_classes_idx_updated; 
    Mode = 1;    
    number_classes_all = length(classes_data)+1;
    classes_to_plot = [1:length(classes_data)]; 
    %before clustering
    figure
    axis([ 134 142 100 125]) %err cases
    colors = colormap(colorcube(number_classes_all));
    hold on
    for i = 1:length(classes_to_plot)
        class_i = classes_data{classes_to_plot(i)}(1:end,:);
        if ~isempty(class_i)
            x = class_i(:,2);
            y = class_i(:,3);
            if Mode == 1
                plot(x, y, 'o', ...
                    'Color', colors(i,:), ... 
                    'MarkerFaceColor', colors(i,:), ...
                    'DisplayName', sprintf('class: %d',classes_to_plot(i)));
            else
                plot(x, y, 'o', ...
                'Color', colors(i,:), ... 
                'DisplayName', sprintf('class: %d',classes_to_plot(i)));
            end
        end
    end
%    legend('show','Location','northeastoutside'); 
%     saveas(gcf, ['~/ThesisNam/WorkSpace/NewYear/figures/' 'm' 'Ori' '.png'])

%save('CLUSTER_CLT/result/whole_NN_K=3,3')
