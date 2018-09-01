import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ImageDataFetcher extends BaseDataFetcher {
    private int height, width;
    private List<double[]> inputs, outputs;
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

    public void addStringData(List<String> paths, List<double[]> outputs) throws InterruptedException {
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

    public void addImageData(List<BufferedImage> input, List<double[]> output) throws InterruptedException {
        List<double[]> imgData = convertData(input);
        List<Integer> wrongIndexes = new ArrayList<>();
        for (int i = 0; i < imgData.size(); i++) {
            if (imgData.get(i) == null)
                wrongIndexes.add(i);
        }
        List<double[]> newInputs = new ArrayList<>(), newOuts = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            if (wrongIndexes.contains(i)){continue;}
            newInputs.add(imgData.get(i));
            newOuts.add(output.get(i));
        }
        inputs.addAll(newInputs);
        outputs.addAll(newOuts);
        this.totalExamples = inputs.size();
    }

    public List<double[]> convertData(List<BufferedImage> input) throws InterruptedException {
        List<double[]> ret = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            ret.add(extractImage(input.get(i)));
        }
        return ret;
    }

    public double[] extractImage(BufferedImage image) throws InterruptedException {
        try {
            BufferedImage dimg = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            Graphics2D g = dimg.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(image, 0, 0, width, height, 0, 0, image.getWidth(), image.getHeight(), null);
            g.dispose();
            int[] prePixels = new int[height*width];
            PixelGrabber pgb = new PixelGrabber(dimg, 0, 0, width, height, prePixels, 0, width);
            pgb.grabPixels();
            double[] pixels = new double[height*width];
            return pixels;
        }catch (Exception e){return null;}

    }

    @Override
    public void fetch(int amount) {
        if (!this.hasMore()){
            throw new IllegalStateException("Unable to get more; there are no more images");
        }
        double[][] inps = new double[amount][this.inputColumns];
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
