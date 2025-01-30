% example script to load and manipulate triad

% load triad
ani.figure = open('triad.fig');
ani.triad = [ani.figure.Children.Children];

hx = line([0,1.1],[0,0],[0,0],'color',[0.5,0,0],'LineStyle','--');
hy = line([0,0],[0,-1.1],[0,0],'color',[0,0.5,0],'LineStyle','--');
hz = line([0,0],[0,0],[0,-1.1],'color',[0,0,0.5],'LineStyle','--');
ani.triadRef = [hx,hy,hz];

% example to manipulate triad
rotate(ani.triad,[0 0 1], 30,[0,0,0]);

xlim([-1 1]*1.1)
ylim([-1 1]*1.1)
zlim([-1 1]*1.1)
axis on

