%Script to evaluate results on the HW1_dataset
%
%Parameters to be changed:
datasetPath='../HW1_dataset/';     %path to the dataset
binaryRootPath='../results/';      %path to the results

%anything below here needs to be changed
categoryList = filesys('getFolders', binaryRootPath);

datasetStats = [];
evaluatedCategories = [];

    
        categoryStats = [];
        names = [];

            videoPath = datasetPath;
            binaryPath = binaryRootPath;
            if exist(binaryPath, 'dir') 
                confusionMatrix = processVideoFolder(videoPath, binaryPath);
                [TP, FP, FN, TN, stats] = confusionMatrixToVar(confusionMatrix);
                categoryStats = [categoryStats; stats];
                nome='      ';
                %nome(1:min(6,length(video)))=video(1:min(6,length(video)));
                names = [names; nome];
            else
                fprintf('ERROR: missing directory for video %s\n',video);
            end

        fprintf('\nName\tRecall\t\tSpecificity\tFPR\t\tFNR\t\tPWC\t\tPrecision\tFMeasure\n');
        for j=1:(size(names,1))
            fprintf('%s\t%1.5f\t%1.5f\t%1.5f\t%1.5f\t%1.5f\t%1.5f\t%1.5f\n', names(j,:),categoryStats(j,:));
        end
        meancategoryStats=mean(categoryStats,1);

        datasetStats = [datasetStats;meancategoryStats];




