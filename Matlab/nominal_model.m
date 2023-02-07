% Calculation of the nominal model for gravity compensation of the
% upper-limb exoskeleton. Fisrt, calculating the transformation matrices
% which were tested to be correct in Trans_Test.m. Then we converted to COM
% and ued euler-langransian method and calculated the potential energy.

clc
clear
close all

%% initial parameters

st=0.005;
p=pi;
Vin=5;
Rref=27.4;
T=8;
min1=2000;
max1=350;

%% DH parameters


syms q1 q2 q3 q4
syms dq1 dq2 dq3 dq4
syms ddq1 ddq2 ddq3 ddq4

q = [pi/2-q1 q2 pi/2+q3 0];
dq = [dq1 dq2 dq3 0];
ddq = [ddq1 ddq2 ddq3 0];

d1 = 0.155 - 0.055;
d2 = 0.105 + 0.055;
d3 = 0.055 + 0.06;
d4 = 0.195 + 0.08;


a3 = 0.275;


DOF = 4;

DH  = sym(zeros(DOF,4));

DH(1,1) = -pi/2;
DH(2,1) = pi/2;
DH(4,1) = pi/2;

DH(3,2) = a3;


% be carefull about sign of the di since it is (Xi)-(Xi-1)
DH(1,3) = d1;
DH(2,3) = d2;
DH(3,3) = -d3;
DH(4,3) = d4;

for i = 1:DOF
    DH(i,4) = q(i);
end

%% calculating the transformation matrix 

T = sym(zeros(4,4,DOF));

% Standard DH parameters
% for i=1:DOF
%     T(:,:,i)=[cos(DH(i,4)) -sin(DH(i,4))*cos(DH(i,1)) sin(DH(i,4))*sin(DH(i,1))  DH(i,2)*cos(DH(i,4)); 
%               sin(DH(i,4))  cos(DH(i,4))*cos(DH(i,1)) -cos(DH(i,4))*sin(DH(i,1)) DH(i,2)*sin(DH(i,4));
%               0             sin(DH(i,1))               cos(DH(i,1))                 DH(i,3);
%               0             0                           0                           1];
% end

% Modified DH parameters
for i=1:DOF
    T(:,:,i)=[cos(DH(i,4))               -sin(DH(i,4))              0                           DH(i,2); 
              sin(DH(i,4))*cos(DH(i,1))  cos(DH(i,4))*cos(DH(i,1))  -sin(DH(i,1))               -DH(i,3)*sin(DH(i,1));
              sin(DH(i,4))*sin(DH(i,1))  cos(DH(i,4))*sin(DH(i,1))  cos(DH(i,1))               DH(i,3)*cos(DH(i,1));
              0                          0                           0                          1];
end

Tb = sym(zeros (4,4,DOF));
Tb(:,:,1) = T(:,:,1);
for i = 2:DOF
    Tb(:,:,i) = Tb(:,:,i-1)*T(:,:,i);
end



%% COM

com = sym(zeros(4,4,DOF-1));


c1 = [1 0 0 0
      0 1 0 -(0.16-(0.045+0.0314))
      0 0 1 -0.1027
      0 0 0 1];
com(:,:,1) = Tb(:,:,1)*c1;

c2 = [1 0 0 0.115;
      0 1 0 0;
      0 0 1 -0.04;
      0 0 0 1;];
com(:,:,2) = Tb(:,:,2)*c2;

c3 = [1 0 0 0;
      0 1 0 -0.1;
      0 0 1 0.04;
      0 0 0 1;];
com(:,:,3) = Tb(:,:,3)*c3;



%% only potential energy

%u = -m*g*h

u = sym(zeros(DOF-1,1));
m = [0.6 0.8 0.4];
g = [0 0 -9.8];

href = 0.55; 

for i = 1:DOF-1
    u(i) = -m(i)*(g*com(1:3, 4, i));
end

U = sum(u);
 

%% Euler-Lagrange eqn.

% L   = KinEn - Pot;
% tao = -dL/dq
% L   = -U;


tao = [diff(U,q1); diff(U,q2); diff(U,q3);]



%% plot

% the relation between q1 and tao1

theta1 = -pi:0.01:pi;
theta2 = 0;
theta3 = 0;

f = subs(tao(1),q2,theta2);
f = subs(f,q3,theta3);
f = subs(f,q1,theta1);
minf1 = double(min(f))- 1.7950;maxf1 = double(max(f))- 1.7950;
f = subs(tao(1),q2,theta2);
f = subs(f,q3,theta3) - 1.7950;
indmin1 = solve(f-minf1,q1); % 
indmax1 = solve(f-maxf1,q1); % 1.0684 (rad), 61.2148(deg)
f1 = f;
f = subs(f,q1,theta1);
% f = f - minf;



subplot(1,3,1)
plot(theta1,f)
xlabel('angle of shAA (rad)')
grid on


% the relation between q2 and tao2

theta1 = 0;
theta2 = -pi:0.01:pi;
theta3 = 0;

f = subs(tao(2),q1,theta1);
f = subs(f,q3,theta3);
f = subs(f,q2,theta2);
minf2 = double(min(f));maxf2 = double(max(f));
f = subs(tao(2),q1,theta1);
f = subs(f,q3,theta3);
indmin2 = solve(f-minf2,q2);
indmax2 = solve(f-maxf2,q2);
f2 = f;
f = subs(f,q2,theta2);
% f = f - minf;

subplot(1,3,2)
plot(theta2,f)
xlabel('angle of shFE (rad)')
grid on


% the relation between q3 and tao3

theta1 = 0;
theta2 = 0;
theta3 = -pi:0.01:pi;

f = subs(tao(3),q1,theta1);
f = subs(f,q2,theta2);
f = subs(f,q3,theta3);
minf3 = double(min(f));maxf3 = double(max(f));
f = subs(tao(3),q1,theta1);
f = subs(f,q2,theta2);
indmin3 = solve(f-minf3,q3);
indmax3 = solve(f-maxf3,q3);
f3 = f;
f = subs(f,q3,theta3);
% f = f - minf;


subplot(1,3,3)
plot(theta3,f)
xlabel('angle of Elbow (rad)')
grid on

%% tao shift

% tao = [diff(U,q1); diff(U,q2); diff(U,q3);];
% 
% 
% % the relation between q1 and tao1
% 
% theta1 = -pi:0.01:pi;
% theta2 = 0;
% theta3 = 0;
% 
% f = subs(tao(1),q2,theta2);
% f = subs(f,q3,theta3);
% f = subs(f,q1,theta1);
% minf = min(f);maxf = max(f);
% f = f(120:450);
% f = f+minf;
% theta1 = theta1(316:316+size(f,2));
% 
% 
% figure(2)
% subplot(1,3,1)
% plot(theta1,f)
% grid on
% 
% 
% % the relation between q2 and tao2
% 
% theta1 = 0;
% theta2 = -pi:0.01:pi;
% theta3 = 0;
% 
% f = subs(tao(2),q1,theta1);
% f = subs(f,q3,theta3);
% f = subs(f,q2,theta2);
% 
% subplot(1,3,2)
% plot(theta2,f)
% grid on
% 
% 
% % the relation between q3 and tao3
% 
% theta1 = 0;
% theta2 = 0;
% theta3 = -pi:0.01:pi;
% 
% f = subs(tao(3),q1,theta1);
% f = subs(f,q2,theta2);
% f = subs(f,q3,theta3);
% 
% subplot(1,3,3)
% plot(theta3,f)
% grid on

%% conversion to pwm

% for elbow
syms u1
func = (-384.08*((((u1*14.5038))/90))^4 + 789.52*((((u1*14.5038))/90))^3 - 618.81*((((u1*14.5038))/90))^2 + 468.18*((((u1*14.5038))/90)));

u2 = 0:0.1:10;
func2 = subs(func,u1,u2);
figure(3)
plot(u2,func2)







