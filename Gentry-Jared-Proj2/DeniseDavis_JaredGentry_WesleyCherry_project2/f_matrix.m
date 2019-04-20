%copyright Denise S Davis, M.S.  4/16/2019
%%  Computing E and F
%%
load('rightcamera.txt')
load('leftcamera.txt')
load('leftmarks.txt')
load('rightmarks.txt')

l2d=leftcamera(:,2:3);
r2d=rightcamera(:,2:3);

z=repmat([0], 1, 3);
A = repmat([0], 1, 9);

%
dr=r2d;
dl=l2d;

%
% train model : build projection matrix
%
for n = 1:size(dr, 1)
    xl=dl(n,1);
    xr=dr(n,1);
    yr=dr(n,2);
    yl=dl(n,2);
    p=[xl*xr xl*yr xl yl*xr yl*yr yl xr yr 1];
    A=[A; p];
end

A(1,:)=[];

[U,S,V]=svd(A);
vp=(V(:,9))
nu=A*vp;
F=[vp(1) vp(2) vp(3); vp(4) vp(5) vp(6); vp(7) vp(8) vp(9)]

[Uf, Df, Vf]=svd(F)
% set singularity to zero
Df(3,3)=0

Fp=Uf*Df*Vf'

%Wl=[402.627 0 181.575; 0 397.591 118.957; 0 0 1]
%Wr=[413.097 0 184.373; 0 409.626 129.361; 0 0 1]
Wl=[401.772 0 184.15; 0 396.122 118.146; 0 0 1]
Wr=[412.208 0 187.93; 0 407.788 131.291; 0 0 1]
E=Wl'*Fp*Wr
%%


%%
[Fpo,inliers] = estimateFundamentalMatrix(l2d,r2d,'NumTrials',4000);
%%
%Overlays two images
I1 = imread('leftface.jpg');
I2 = imread('rightface.jpg');
figure;
ax=axes;
showMatchedFeatures(I1, I2, leftmarks(inliers2,:),rightmarks(inliers2,:), 'Parent', ax);
title('Matched Points Overlay(Before Rectification) Inliers');

%%
I1 = imread('newleft.png');
I2 = imread('newright.png');
figure;
ax=axes;
showMatchedFeatures(I1, I2, newleftmarks, ...
    newrightmarks, 'Parent', ax);
title('Matched Points Overlay(Rectification)');
%%
I1 = imread('newleft.png');
I2 = imread('newright.png');
figure;
ax=axes;
showMatchedFeatures(I1, I2, newleftmarks, ...
    newrightmarks,'montage','Parent', ax);
title('Matched Points (After rectification) Inliers');
%%
I1 = imread('newleft.png');
I2 = imread('newright.png');
figure
subplot(121);
imshow(I1); 
title('Marks and Epipolar Lines in Left Image'); hold on;
plot(newleftmarks(:,1), newleftmarks(:,2),'g+')
epiLines = epipolarLine(Fp', newrightmarks(:,:));
points = lineToBorderPoints(epiLines,size(I1));
line(points(:,[1,3])',points(:,[2,4])');


subplot(122);
imshow(I2);
title('Marks and Epipolar Lines in Right Image'); hold on;
plot(newrightmarks(:,1), newrightmarks(:,2),'g+')

epiLines_2 = epipolarLine(Fp', newleftmarks(:,:));
points_2 = lineToBorderPoints(epiLines_2,size(I2));
line(points_2(:,[1,3])',points_2(:,[2,4])');

%%
I1 = imread('leftface.jpg');
I2 = imread('rightface.jpg');
subplot(121);
imshow(I1); 
title('Marks and Epipolar Lines in Left Image'); hold on;
plot(leftmarks(:,1), leftmarks(:,2),'g+')
epiLines = epipolarLine(Fp', rightmarks(inliers2,:));
points = lineToBorderPoints(epiLines,size(I1));
line(points(:,[1,3])',points(:,[2,4])');


subplot(122);
imshow(I2);
title('Marks and Epipolar Lines in right Image'); hold on;
plot(rightmarks(:,1), rightmarks(:,2),'g+')

epiLines_2 = epipolarLine(Fp', leftmarks(inliers2,:));
points_2 = lineToBorderPoints(epiLines_2,size(I2));
line(points_2(:,[1,3])',points_2(:,[2,4])');