function variation_dist(statistics, m)

figure;
plot(statistics(:, 1), statistics(:, 4), 'b*-', 'lineWidth', 2);
% loglog(statistics(:, 1), statistics(:, 4), 'b*-', 'lineWidth', 2);
hold on;
plot(statistics(:, 1), statistics(:, 5), 'gx-', 'lineWidth', 2);
% loglog(statistics(:, 1), statistics(:, 5), 'gx-', 'lineWidth', 2);
title_content = sprintf('Number of Latent Variable m = %d', m);
title(title_content);
xlabel('Training Size');
ylabel('Variation Distance');
grid on;
legend('LearnHMM', 'EM');