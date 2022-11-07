#!bin/bash

para_file=$1"/col1_para_generated"
non_para_file=$1"/col1_nonpara_generated"
original_badcase=$1"/badcases"

awk 'Begin{FS="\t"; OFS="\t"} $1!=$2{print $1 "\t" $2 "\t" $3}'  $para_file > $para_file'_non_repeat'
awk 'Begin{FS="\t"; OFS="\t"} $1!=$2{print $1 "\t" $2 "\t" $3}'  $non_para_file > $non_para_file'_non_repeat'


# cat $original_badcase $para_file'_non_repeat' $non_para_file'_non_repeat' > $1'/col1_all'

cat $para_file'_non_repeat' $non_para_file'_non_repeat' > $1'/col1_all'


# mv $1'/col_all' $1'/col_all_shuf'
# shuf $1'/col1_all' > $1'/col1_all_shuf'

rm $para_file'_non_repeat'
rm $non_para_file'_non_repeat'
# rm $1'/col1_all'

echo 'finished'