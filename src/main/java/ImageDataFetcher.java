import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ImageDataFetcher extends BaseDataFetcher {
    private int height, width;
    private List<double[][][]> inputs;
    private List<double[]> outputs;
    public ImageDataFetcher(int height, int width, int classesAmount){
        this.cursor = 0;
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
        this.height = height;
        this.width = width;
        this.numOutcomes = classesAmount;
        this.totalExamples = 0;
        this.inputColumns = height*width;
    }

    public void addStringData(List<String> paths, List<double[]> outputs) {
        List<BufferedImage> list = new ArrayList<>();
        for (int i = 0; i < paths.size(); i++) {
            try {
                list.add(ImageIO.read(new File(paths.get(i))));
            } catch (IOException e) {
                System.out.println(paths.get(i));
                e.printStackTrace();
            }
        }
        addImageData(list, outputs);
    }

    public void addImageData(List<BufferedImage> input, List<double[]> output) {
        List<double[][][]> imgData = convertData(input);
        List<Integer> wrongIndexes = new ArrayList<>();
        for (int i = 0; i < imgData.size(); i++) {
            if (imgData.get(i) == null)
                wrongIndexes.add(i);
        }
        List<double[][][]> newInputs = new ArrayList<>();
        List<double[]> newOuts = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            if (wrongIndexes.contains(i)){continue;}
            newInputs.add(imgData.get(i));
            newOuts.add(output.get(i));
        }
        inputs.addAll(newInputs);
        outputs.addAll(newOuts);
        dataStorage<List<double[][][]>, List<double[]>> mixedData = mixData(inputs, outputs);
        inputs = mixedData.getKey();
        outputs = mixedData.getValue();
        this.totalExamples = inputs.size();
    }

    private dataStorage<List<double[][][]>, List<double[]>> mixData(List<double[][][]> inps, List<double[]> outs){
        List<Integer> indexes =new ArrayList<>();
        for (int i = 0; i < inps.size(); i++) { indexes.add(i); }
        Collections.shuffle(indexes);
        List<double[][][]> mixedInps = new ArrayList<>();
        List<double[]> mixedOuts = new ArrayList<>();
        for(int index: indexes){
            mixedInps.add(inps.get(index).clone());
            mixedOuts.add(outs.get(index).clone());
        }
        return new dataStorage<>(mixedInps, mixedOuts);
    }

    public List<double[][][]> convertData(List<BufferedImage> input) {
        List<double[][][]> ret = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            ret.add(extractImage(input.get(i)));
        }
        return ret;
    }

    public double[][][] extractImage(BufferedImage image) {
        try {
            BufferedImage dimg = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = dimg.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(image, 0, 0, width, height, 0, 0, image.getWidth(), image.getHeight(), null);
            g.dispose();
            int[] prePixels = dimg.getRGB(0, 0, width, height, null, 0, width);

            double[][][] pixels = new double[3][height][width];
            int counter = 0;
            for(int item: prePixels){
                Color a =new Color(item);
                pixels[0][counter%width][(counter-counter%width)/width] = a.getRed()/3-85;
                pixels[1][counter%width][(counter-counter%width)/width] = a.getGreen()/3-85;
                pixels[2][counter%width][(counter-counter%width)/width] = a.getBlue()/3-85;
                counter++;
            }
            return pixels;
        }catch (Exception e){return null;}

    }

    @Override
    public void fetch(int amount) {
        if (!this.hasMore()){
            throw new IllegalStateException("Unable to get more; there are no more images");
        }
        if (amount>this.totalExamples+cursor){amount = this.totalExamples-cursor;}
        double[][][][] inps = new double[amount][3][height][width];
        double[][] outs = new double[amount][this.numOutcomes];
        for (int i = 0; i < amount && this.hasMore(); i++) {
            inps[i] = inputs.get(this.cursor).clone();
            outs[i] = outputs.get(this.cursor).clone();
            this.cursor++;
        }
        INDArray in = Nd4j.create(inps), out  = Nd4j.create(outs);
        this.curr = new DataSet(in, out);
    }
}

class dataStorage<K, M>{
    private K key;
    private M value;
    public dataStorage(K key, M value) {
        this.key = key;
        this.value = value;
    }
    public K getKey(){return key;}
    public M getValue(){return value;}
}
