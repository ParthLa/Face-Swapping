function [ left_eye, right_eye , Face, imgFace] = getLandmarks( I )
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

% I = face_warp;
[imgFace, LeftEye, RightEye, Mouth, Face] = detectFacialRegions(I);

% config landmarks to Eyes and Mouth (4 and 5)
landconf = 5;


%% landmarks the eyes
imgLeftEye = (imgFace(LeftEye(1,2):LeftEye(1,2)+LeftEye(1,4),LeftEye(1,1):LeftEye(1,1)+LeftEye(1,3),:));
[landLeftEye, leftEyeCont] = eyesProcessing(imgLeftEye,landconf);

imgRightEye = (imgFace(RightEye(1,2):RightEye(1,2)+RightEye(1,4),RightEye(1,1):RightEye(1,1)+RightEye(1,3),:));
[landRightEye, rightEyeCont] = eyesProcessing(imgRightEye,landconf);


%% shows (eyes and mouth)

figure; imshow(imgFace,'InitialMagnification',50); hold on;
[left_eye_x, left_eye_y]=showsLandmarks(landLeftEye,leftEyeCont,LeftEye,landconf);
[right_eye_x, right_eye_y]=showsLandmarks(landRightEye,rightEyeCont,RightEye,landconf);

if left_eye_x <= right_eye_x
    left_eye = [left_eye_x, left_eye_y];
    right_eye = [right_eye_x, right_eye_y];
else
    left_eye = [right_eye_x, right_eye_y];
    right_eye = [left_eye_x, left_eye_y];
end


