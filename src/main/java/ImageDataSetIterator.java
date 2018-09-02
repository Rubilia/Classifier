import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

import java.awt.image.BufferedImage;
import java.util.List;

public class ImageDataSetIterator extends BaseDatasetIterator {
    private static ImageDataFetcher fetcher;
    public ImageDataSetIterator(int batch, int h, int w, int classesAmount){
        this(batch, 0, new ImageDataFetcher(h, w, classesAmount));

    }

    public ImageDataSetIterator(int batch, int numExamples, ImageDataFetcher fetcher) {
        super(batch, numExamples, fetcher);
        this.fetcher = fetcher;
    }

    public void addDataString(List<String> paths, List<double[]> out) {
        fetcher.addStringData(paths, out);
        super.numExamples = fetcher.totalExamples();
    }

    public void addDataImage(List<BufferedImage> paths, List<double[]> out) {
        fetcher.addImageData(paths, out);
        super.numExamples = fetcher.totalExamples();
    }
}
