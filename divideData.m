
%Divide Data
%Parameters£º
% -outcome: (0,1) of each patient.
% -cvind:  CV indices of each image files.
clc;clear;
label=[ones(35,1)-2;zeros(21,1)+1;zeros(20,1)+1];
clear inflam recurrence recurrenceH;
nPatient = numel(label);

[train, test]=crossvalind('HoldOut', label, 0.3);   %test: nPatient*0.3
save data_index train test;

