

public class Simulator extends Thread {

    private static void oneRound(String DEFAULT_IP, int DEFAULT_PORT, int DEFAULT_TIMEOUT, int K) {

        for (int i = 0; i < K; i++) {
            FileClient object = FileClient.connect(DEFAULT_IP, DEFAULT_PORT, DEFAULT_TIMEOUT);
            object.start();
        }
    }

    public static void main(String[] args) throws InterruptedException {

//        String DEFAULT_IP = "localhost";
        String DEFAULT_IP = "localhost";
        int DEFAULT_PORT = 8000;
        int DEFAULT_TIMEOUT = 5000;
        int K = 10;
        int round = 10;

        for (int r = 0; r < round; r++) {
            System.out.println("round: " + r);
            oneRound(DEFAULT_IP, DEFAULT_PORT, DEFAULT_TIMEOUT, K);
            Thread.sleep(60 * 1000);
        }

    }

}

