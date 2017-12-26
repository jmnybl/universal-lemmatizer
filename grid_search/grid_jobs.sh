
# parameters: treebank code
tb=$1


### run grid

# (batch_size, dropout, artificial data, copy attention)

for batch in 6 12 36
do
    for drop in 0.0 0.05 0.1
    do
        
        for art in 5000 10000 30000 100000
        do

            for copy in "false" "true"
            do

                echo "./train_lang_arg.sh --treebank=$tb --batchsize=$batch --dropout=$drop --epochs=20 --gpuid=0 --artificial_data=$art --copy_attention=$copy" 

            done

        done

    done

done


