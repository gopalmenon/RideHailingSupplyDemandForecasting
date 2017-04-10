load('DistrictsViz.txt');load('TimeslotsViz.txt');load('RidesViz.txt');scatter3(DistrictsViz, TimeslotsViz, RidesViz,[], RidesViz);
title("Number of Rides for Start District and Time Slot");
xlabel("District");
ylabel("Time Slot");