import org.nd4j.jita.conf.CudaEnvironment;

public class Main {
    public static void main(String[] args) {
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMaximumDeviceCache(2L * 1024L * 1024L * 1024L).allowCrossDeviceAccess(true);

    }
}
