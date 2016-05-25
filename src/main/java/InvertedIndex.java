import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

public class InvertedIndex {
    /**
     * Mapper部分
     **/
    public static class InvertedIndexMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);//常量1

        /**
         * 对输入的Text切分为多个word,每个word作为一个key输出
         * 输入：key:当前行偏移位置, value:当前行内容
         * 输出：key:word#filename, value:1
         */
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            String fileName = fileSplit.getPath().getName().toLowerCase();//获取文件名，为简化下一步对文件名处理，转换为小写
            int pos = fileName.indexOf(".");
            if (pos > 0) {
                fileName = fileName.substring(0, pos);//去除文件名后缀
            }
            Text word = new Text();
            StringTokenizer itr = new StringTokenizer(value.toString());
            for (; itr.hasMoreTokens(); ) {
                word.set(itr.nextToken() + "#" + fileName);
                context.write(word, one);//输出 word#filename 1
            }
        }
    }

    /**
     * Combiner部分
     **/
    public static class SumCombiner extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();

        /**
         * 将Mapper输出的中间结果相同key部分的value累加，减少向Reduce节点传输的数据量
         * 输出：key:word#filename, value:累加和
         */
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    /**
     * Partitioner部分
     **/
    public static class NewPartitioner extends HashPartitioner<Text, IntWritable> {
        /**
         * 为了将同一个word的键值对发送到同一个Reduce节点，对key进行临时处理
         * 将原key的(word, filename)临时拆开，使Partitioner只按照word值进行选择Reduce节点
         */
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            String term = key.toString().split("#")[0];//获取word#filename中的word
            return super.getPartition(new Text(term), value, numReduceTasks);
        }
    }

    /**
     * Reducer部分
     **/
    public static class InvertedIndexReducer extends Reducer<Text, IntWritable, Text, Text> {
        private String term = new String();//临时存储word#filename中的word
        private String last = " ";//临时存储上一个word
        private int countItem;//统计word出现次数
        private int countDoc;//统计word出现文件数
        private StringBuilder out = new StringBuilder();//临时存储输出的value部分
        private float f;//临时计算平均出现频率

        /**
         * 利用每个Reducer接收到的键值对中，word使排好序的
         * 只需要将相同的word中，将(word,filename)拆分开，将filename与累加和拼到一起，存储到临时StringBuilder中
         * 待出现word不同，则将此word作为key，存储有此word出现的全部filename及其出现次数的StringBuilder作为value输出
         * 在处理相同word的同时，还会统计其出现的文件数目，总的出现次数，以此计算其平均出现频率
         * 输入：key:word#filename, value:[NUM,NUM,...]
         * 输出：key:word, value:平均出现频率,filename:NUM;filename:NUM;...
         */
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            term = key.toString().split("#")[0];//获取word
            if (!term.equals(last)) {//此次word与上次不一样，则将上次进行处理并输出
                if (!last.equals(" ")) {//避免第一次比较时出错
                    out.setLength(out.length() - 1);//删除value部分最后的;符号
                    f = (float) countItem / countDoc;//计算平均出现次数
                    context.write(new Text(last), new Text(String.format("%.2f,%s", f, out.toString())));//value部分拼接后输出
                    countItem = 0;//以下清除变量，初始化计算下一个word
                    countDoc = 0;
                    out = new StringBuilder();
                }
                last = term;//更新word，为下一次做准备
            }
            int sum = 0;//累加相同word和filename中出现次数
            for (IntWritable val : values) {
                sum += val.get();
            }
            out.append(key.toString().split("#")[1] + ":" + sum + ";");//将filename:NUM; 临时存储
            countItem += sum;
            countDoc += 1;
        }

        /**
         * 上述reduce()只会在遇到新word时，处理并输出前一个word，故对于最后一个word还需要额外的处理
         * 重载cleanup()，处理最后一个word并输出
         */
        public void cleanup(Context context) throws IOException, InterruptedException {
            out.setLength(out.length() - 1);
            f = (float) countItem / countDoc;
            context.write(new Text(last), new Text(String.format("%.2f,%s", f, out.toString())));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Job job = new Job(conf, "inverted index");
        job.setJarByClass(InvertedIndex.class);
        job.setMapperClass(InvertedIndexMapper.class);
        job.setCombinerClass(SumCombiner.class);
        job.setReducerClass(InvertedIndexReducer.class);
        job.setNumReduceTasks(5);//设定使用Reduce节点个数
        job.setPartitionerClass(NewPartitioner.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
