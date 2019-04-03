function [ methodID ] = readInput( list )
%READINPUT  

fprintf('Please, select a  method from the list:\n'); 
for i=1:length(list)
   fprintf('[%d] %s \n',i,list{i});
end
methodID = input('> ');

end

