statistics = load('m4n8_SmallSynV_COMP.txt');
statistics = statistics(1:end-1, :);
rank_hyperparameter = 4;
training_size = statistics(:, 1);
sl_variation_dist = statistics(:, 4);
em_variation_dists = statistics(:, 5:end);
mean_em_variation_dist = mean(em_variation_dists, 2);
std_em_variation_dist = std(em_variation_dists, 0, 2);

figure;
hold on;
plot(training_size, sl_variation_dist, 'b-*', 'lineWidth', 2);
errorbar(training_size, mean_em_variation_dist, std_em_variation_dist, ...
        'r-x', 'lineWidth', 2);
hold off;
grid on;
legend('LearnHMM', 'EM');
xlabel('Training Size', 'fontsize', 15);
ylabel('Variation Distance', 'fontsize', 15);
title_content = sprintf('SmallSynV, m = %d', rank_hyperparameter);
title(title_content, 'fontsize', 15);