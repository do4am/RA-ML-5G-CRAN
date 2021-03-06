% EXAMPLE RUNING MERGER with Scatterer's density = 0.01 / m^2 dataset
% Load S1-library.mat and S1-pos.mat
[classes_library_updated2, classes_sorted_idx2, log_changed_1] = ... 
    cl_merger(classes_library_updated, ...
    classes_sorted, classes_sorted_idx);

% [classes2_lib_updated2, classes2_sorted_idx2, log_changed_1] = ... 
%     cl_merger(classes2_lib_updated, ...
%     classes2_sorted, classes2_sorted_idx);

% [toy_classes_lib2, toy_classes_idx2, log_changed_1] = ... 
%     cl_merger(toy_classes_lib, ...
%     toy_classes, toy_classes_idx);
%% remove empty classes
% Class varnished (Class size = 0) will be removed from classes_list. This is an option.
count = 0;
classes_sorted_idx_final = classes2_sorted_idx2;%classes2_sorted_idx2;%toy_classes_idx2;
classes_sorted_final = classes2_sorted;%classes2_sorted;%toy_classes;

number_classes = length(classes_sorted_idx_final);
i = 1;

while i <= number_classes
    if ~isempty(classes_sorted_idx_final{i})
        count = count+1;
    else
        classes_sorted_idx_final(i) = [];
        classes_sorted_final(i,:) = [];
        i = i-1;
    end
    i = i+1;
    number_classes = length(classes_sorted_idx_final);
end

[classes_sorted_final, classes_sorted_idx_final] = cl_sort(classes_sorted_final, classes_sorted_idx_final);

%% Plotting Toy Merged
%Ploting scattering plot. classes2_sorted_idx2 (to match the color with the original before clustering) or classes_sorted_final. 
    classes_data = classes2_sorted_idx2; %classes2_sorted_idx2;%classes_sorted_idx2;
                                        %toy_classes_idx2; %toy_classes_idx_updated; 
                                        
    Mode = 1;  % an option to color the dot
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
    %legend('show','Location','northeastoutside'); 