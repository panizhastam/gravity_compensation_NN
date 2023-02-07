clc
clear
close all

load('G:\My Drive\SoftExo\Temp\GCdata\alljoints2.mat')

t = data{1}.Values.Time;

Elbow_tao = data{1}.Values.Data;
ShAA_tao = data{2}.Values.Data;
ShFE_tao = data{3}.Values.Data;
ShFE_theta = data{4}.Values.Data;
ShAA_theta = data{6}.Values.Data;
Elbow_theta = data{7}.Values.Data;

plot(t,Elbow_tao,t,Elbow_theta)
title("Elbow data")
xlabel('t')
legend('tao','theta')



%% Elbow data
% input
Elbow_theta_up = [Elbow_theta(1:786); Elbow_theta(786*2:end)];
Elbow_theta_down = [Elbow_theta(786:2*786); Elbow_theta(786*3:786*4)];
% output
Elbow_tao_up = [Elbow_tao(1:786); Elbow_tao(786*2:786*3); Elbow_tao(786*4:end)];
Elbow_tao_down = [Elbow_tao(786:2*786); Elbow_tao(786*3:786*4)];


%% ShFE data
% input
ShFE_theta_up = [ShFE_theta(1:786); ShFE_theta(786*2:786*3); ShFE_theta(786*4:end)];
ShFE_theta_down = [ShFE_theta(786:2*786); ShFE_theta(786*3:786*4)];
% output
ShFE_tao_up = [ShFE_tao(1:786); ShFE_tao(786*2:786*3); ShFE_tao(786*4:end)];
ShFE_tao_down = [ShFE_tao(786:2*786); ShFE_tao(786*3:786*4)];


%% ShAA data
% input
ShAA_theta_up = [ShAA_theta(1:786); ShAA_theta(786*2:786*3); ShAA_theta(786*4:end)];
ShAA_theta_down = [ShAA_theta(786:2*786); ShAA_theta(786*3:786*4)];
% output
ShAA_tao_up = [ShAA_tao(1:786); ShAA_tao(786*2:786*3); ShAA_tao(786*4:end)];
ShAA_tao_down = [ShAA_tao(786:2*786); ShAA_tao(786*3:786*4)];


%% saving data to .mat file

MyData_up = [Elbow_theta_up ShFE_theta_up ShAA_theta_up...
            Elbow_tao_up ShFE_tao_up ShAA_tao_up];

MyData_down = [Elbow_theta_down ShFE_theta_down ShAA_theta_down...
            Elbow_tao_down ShFE_tao_down ShAA_tao_down];

% 
% save('alljoints2data_up.mat','MyData_up');
% save('alljoints2data_down.mat','MyData_down');



%% Merge and save as .csv

% d1 = load("Elbowdata_up.mat").MyData_up;
% d2 = load("ShFEdata_up.mat").MyData_up;
% d3 = load("ShAAdata_up.mat").MyData_up;
% d4 = load("alljoints1data_up.mat").MyData_up;
% d5 = load("alljoints2data_up.mat").MyData_up;
% 
% newd = [d1;d2;d3;d4;d5];
% save('alldata_up.mat','newd')
% csvwrite('alldata_up.csv',newd)
% 
% d1 = load("Elbowdata_down.mat").MyData_down;
% d2 = load("ShFEdata_down.mat").MyData_down;
% d3 = load("ShAAdata_down.mat").MyData_down;
% d4 = load("alljoints1data_down.mat").MyData_down;
% d5 = load("alljoints2data_down.mat").MyData_down;
% 
% newd = [d1;d2;d3;d4;d5];
% save('alldata_down.mat','newd')
% csvwrite('alldata_down.csv',newd)






