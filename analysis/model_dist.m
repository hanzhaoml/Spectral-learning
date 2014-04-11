% statistics1 = load('./m4n8_new/m4n8_10000_V.txt');
% statistics5 = load('./m4n8_new/m4n8_50000_V.txt');
% statistics10 = load('./m4n8_new/m4n8_100000_V.txt');
% statistics50 = load('./m4n8_new/m4n8_500000_V.txt');
% statistics100 = load('./m4n8_new/m4n8_1000000_V.txt');
% 
statistics1 = load('./m50n100_new/m50n100_10000_F.txt');
statistics5 = load('./m50n100_new/m50n100_50000_F.txt');
statistics10 = load('./m50n100_new/m50n100_100000_F.txt');
statistics50 = load('./m50n100_new/m50n100_500000_F.txt');
statistics100 = load('./m50n100_new/m50n100_1000000_F.txt');
m = 50;
M = 1:50;
statistics1 = statistics1(M, :);
statistics5 = statistics5(M, :);
statistics10 = statistics10(M, :);
statistics50 = statistics50(M, :);
statistics100 = statistics100(M, :);

figure;
hold on;
plot(M, statistics1(:, 1), 'b-', 'lineWidth', 2);
plot(M, statistics5(:, 1), 'g-', 'lineWidth', 2);
plot(M, statistics10(:, 1), 'r-', 'lineWidth', 2);
plot(M, statistics50(:, 1), 'k-', 'lineWidth', 2);
plot(M, statistics100(:, 1), 'm-', 'lineWidth', 2);

% plot(M, statistics1(:, 1), 'b-*', 'lineWidth', 2);
% plot(M, statistics5(:, 1), 'g-x', 'lineWidth', 2);
% plot(M, statistics10(:, 1), 'r-s', 'lineWidth', 2);
% plot(M, statistics50(:, 1), 'k-d', 'lineWidth', 2);
% plot(M, statistics100(:, 1), 'm-+', 'lineWidth', 2);
set(gca, 'yscale', 'log');

xlabel('Rank Hyperparameter', 'fontsize', 15);
ylabel('L1 error', 'fontsize', 15);
title('LargeSyn', 'fontsize', 15);
hold off;
grid on;
leg_handle = legend('Training Size = 10000', ...
       'Training Size = 50000', ...
       'Training Size = 100000', ... 
       'Training Size = 500000', ...
       'Training Size = 1000000');
set(leg_handle, 'fontsize', 8);
   
   
figure;
hold on;
plot(M, statistics1(:, 3), 'b-', 'lineWidth', 2);
plot(M, statistics5(:, 3), 'g-', 'lineWidth', 2);
plot(M, statistics10(:, 3), 'r-', 'lineWidth', 2);
plot(M, statistics50(:, 3), 'k-', 'lineWidth', 2);
plot(M, statistics100(:, 3), 'm-', 'lineWidth', 2);

% plot(M, statistics1(:, 3), 'b-*', 'lineWidth', 2);
% plot(M, statistics5(:, 3), 'g-x', 'lineWidth', 2);
% plot(M, statistics10(:, 3), 'r-s', 'lineWidth', 2);
% plot(M, statistics50(:, 3), 'k-d', 'lineWidth', 2);
% plot(M, statistics100(:, 3), 'm-+', 'lineWidth', 2);

xlabel('Rank Hyperparameter', 'fontsize', 15);
ylabel('Proportion of Negative Probabilities', 'fontsize', 15);
title('LargeSyn', 'fontsize', 15);
hold off;
grid on;
leg_handle = legend('Training Size = 10000', ...
       'Training Size = 50000', ...
       'Training Size = 100000', ... 
       'Training Size = 500000', ...
       'Training Size = 1000000');
set(leg_handle, 'fontsize', 8);