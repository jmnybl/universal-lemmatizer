# usage: ./convert_all_transducers.sh input_directory output_directory


input=$1
output=$2

mkdir -p $output


for f in $input/* ; do
    echo "converting" $f
    base=$(basename $f)
#    base=${base%.*}
    zcat $f | python3 convert_to_ud.py -f apertium --feature_mapping apertium2ud.tsv | gzip -c > $output/$base
done
