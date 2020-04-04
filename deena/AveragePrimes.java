public class AveragePrimes {
    public static void main(String[] inputs) {
        int n = Integer.parseInt(inputs[0]);
        if (n<2) {
            System.out.println("Invalid value");
            return;
        }
        boolean[] nonprimes = new boolean[n-1];
        for(int d=2; d <= Math.round(Math.sqrt(n)); d++) {
            for(int i=d*2; i <= n; i += d) {
                //System.out.println(String.valueOf(i));
                nonprimes[i-2]=true;
            }
        }
        int primeCounter=0;
        int primeTotal=0;
    
        for(int i=2; i<=n; i++) {
            if(!nonprimes[i-2]) {
                System.out.println(String.valueOf(i));
                primeCounter++;
                primeTotal+=i;
            }
        }

        System.out.println("Hello World! "+ String.valueOf((double)primeTotal/(double)primeCounter));
        
    }
}