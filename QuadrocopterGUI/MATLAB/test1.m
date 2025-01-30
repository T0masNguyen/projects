
Dim_X = 20; %size of the cube in X
Dim_Y = 20; %size of the cube in Y
Dim_Z = 20; %size in Z axes
vertex_matrix = [0 0 0;
1 0 0;
1 1 0;
0 1 0;
0 0 1;
1 0 1;
1 1 1;
0 1 1];
% vertex_matrix = [0 0 0;
% 4 0 0;
% 4 3 0;
% 0 3 0;
% 0 0 1;
% 4 0 1;
% 4 3 1;
% 0 3 1];
 
faces_matrix = [1 2 6 5
2 3 7 6
3 4 8 7
4 1 5 8
1 2 3 4
5 6 7 8];
%origin = CubeCenter;
origin = [0 0 0]; %offset
CubeParameters = [vertex_matrix(:,1)*Dim_X+origin(1),vertex_matrix(:,2)*Dim_Y+origin(2),vertex_matrix(:,3)*Dim_Z+origin(3)];

cube = patch('Vertices',CubeParameters,'Faces',faces_matrix,'FaceColor', 'blue');
s = tic;
xlim([0 100]);
ylim([0 100]);
zlim([0 100]);
while toc(s)< 10
x = randi(100);
y = randi(100);
z = randi(100);
rotate(cube, [1,0,0], x);
rotate(cube, [0,1,0], y);
rotate(cube, [0,0,1], z);
pause(0.1);
view(3);

end
