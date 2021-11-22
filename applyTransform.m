function output_img  = applyTransform(I_d, I_s)

faceDetector = vision.CascadeObjectDetector();

bbox_d = step(faceDetector, I_d);
[left_eye_d,right_eye_d, ~]= getLandmarks(I_d);
[left_eye_s,right_eye_s, Face_s]= getLandmarks(I_s);

theta_d = asin((left_eye_d(2) - right_eye_d(2))/(right_eye_d(1) - left_eye_d(1)));
theta_s = asin((left_eye_s(2) - right_eye_s(2))/(right_eye_s(1) - left_eye_s(1)));

dtheta = theta_s - theta_d;
ds = norm(left_eye_s - right_eye_s) / norm(left_eye_d - right_eye_d);

tform = affine2d([ds*cos(dtheta) -ds*sin(dtheta) 0; ds*sin(dtheta) ds*cos(dtheta) 0; 0 0 1]);

face_warp = imwarp(imcrop(I_d, bbox_d), tform);

lx = ds*(left_eye_d(1)*cos(dtheta) - left_eye_d(2)*sin(dtheta) + max(0,bbox_d(4)*sin(dtheta)));
ly = ds*(left_eye_d(1)*sin(dtheta) + left_eye_d(2)*cos(dtheta) - min(0,bbox_d(3)*sin(dtheta)));
left_eye_w = [lx, ly];

rx = ds*(right_eye_d(1)*cos(dtheta) - right_eye_d(2)*sin(dtheta) + max(0,bbox_d(4)*sin(dtheta)));
ry = ds*(right_eye_d(1)*sin(dtheta) + right_eye_d(2)*cos(dtheta) - min(0,bbox_d(3)*sin(dtheta)));
right_eye_w = [rx, ry];

threshold_xy = round(Face_s(1:2) + 0.5*(left_eye_s - left_eye_w + right_eye_s - right_eye_w));
output_img = uint8(zeros(size(I_s)));

[sz1, sz2, ~] = size(face_warp);
output_img(threshold_xy(1):threshold_xy(1)+sz1 - 1,threshold_xy(2):threshold_xy(2) + sz2 - 1,:)= face_warp;