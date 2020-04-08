import pandas as pd
import sys

sketch_file = sys.argv[1]
output_file = sys.argv[2]
df = pd.read_csv(sketch_file,header=0,sep='\t')
result=df.loc[:,['match','deletion','insertion','mismatch','error rate']].mean(axis=0).to_list()
print(result)
with open(output_file,'w') as f:
    for v in result:
        f.write(str(v)+'\t')