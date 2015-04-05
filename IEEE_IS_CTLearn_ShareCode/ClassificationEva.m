classdef ClassificationEva
    %CLASSIFICATIONEVA Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function eva = ClassificationEva()
        end
       
         % single-label test error
         function error = calTestError(eva, answer, pred)
             error = 0;
             for i=1:length(answer)
                 if answer(i)~=pred(i) 
                     error = error + 1;
                 end
             end
             error = error/ length(answer);
         end

    end
end

