import sys
from BitVector import *
from PrimeGenerator import *

# Homework Number: 6
# Name: Christopher Chan
# ECN Login: chan328   
# Due Date: 2/27/24

class RSA():
    def __init__(self , e) -> None :
        self . e = e
        self . n = None
        self . d = None
        self . p = None
        self . q = None

    # You are free to have other RSA class methods you deem necessary for your solution
    
    def gcd(self, x, y):
        if y == 0:
            return x
        
        return self.gcd(y, x % y)

    def rsaKeyGen(self, pFile, qFile):
        generate = PrimeGenerator(bits = 128)
        p = 0
        q = 0

        while(p == q and self.gcd(p, self.e) != 1 and self.gcd(q, self.e) != 1):
            p = generate.findPrime()
            q = generate.findPrime()

        pWrite = open(pFile, "w")
        qWrite = open(qFile, "w")

        pWrite.write(str(p))
        qWrite.write(str(q))

        pWrite.close()
        qWrite.close()

    def encrypt ( self , plaintext :str , ciphertext :str ) -> None :
        # your implemenation goes here

        pFile = open(sys.argv[3])
        qFile = open(sys.argv[4])

        bv = BitVector(filename = plaintext)
        self.p = pFile.read()
        self.p = int(self.p)
        self.q = qFile.read()
        self.q = int(self.q)
        
        self.n = self.p * self.q    
        output = BitVector(size = 0)

        while(bv.more_to_read):
            bitvec = bv.read_bits_from_file(128)

            #pad with 0s if less than 128 bits
            if bitvec.length() < 128:
                bitvec.pad_from_right(128 - bitvec.length())

            bitvec.pad_from_left(128) #make 256 block by padding 0s from left of 128 block
            output += BitVector(intVal = pow(bitvec.int_val(), self.e, self.n), size=256) # bitvec ^ e % n

        FILEOUT = open(ciphertext, "w")
        FILEOUT.write(output.get_bitvector_in_hex())
        FILEOUT.close()
        
    def decrypt ( self , ciphertext :str , recovered_plaintext :str ) -> None :
        # your implemenation goes here
        
        encryptedText = open(ciphertext)
        pFile = open(sys.argv[3])
        qFile = open(sys.argv[4])

        bv = BitVector(hexstring = encryptedText.read())
        self.p = pFile.read()
        self.p = int(self.p)
        self.q = qFile.read()
        self.q = int(self.q)
        
        self.n = self.p * self.q
        totient = (self.p - 1) * (self.q - 1)
        eBV = BitVector(intVal = self.e)
        dBV = eBV.multiplicative_inverse(BitVector(intVal = totient)) #d x e = 1 mod (p - 1)(q - 1)
        self.d = dBV.int_val()
        output = BitVector(size = 0)

        for i in range(len(bv) // 256):
            bitvec = bv[256*i:256*(i+1)]

            #equations via Lecture 12.5
            vp = pow(bitvec.int_val(), self.d, self.p)
            vq = pow(bitvec.int_val(), self.d, self.q)
            q_1 = BitVector(intVal = self.q).multiplicative_inverse(BitVector(intVal = self.p)) #MI(q, p)
            p_1 = BitVector(intVal = self.p).multiplicative_inverse(BitVector(intVal = self.q)) #MI(p, q)
            xp = self.q * (q_1.int_val() % self.p)
            xq = self.p * (p_1.int_val() % self.q)

            output += BitVector(intVal = (vp * xp + vq * xq) % self.n, size=256)[128:] #remove padding

        FILEOUT = open(recovered_plaintext, "w")
        FILEOUT.write(output.get_text_from_bitvector())
        FILEOUT.close()

if __name__ == "__main__":
    cipher = RSA( e = 65537 )

    if sys.argv[1] == "-g":
        cipher.rsaKeyGen(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "-e":
        cipher.encrypt ( plaintext = sys.argv[2], ciphertext = sys.argv[5])
    elif sys.argv[1] == "-d":
        cipher.decrypt( ciphertext = sys.argv[2], recovered_plaintext = sys.argv[5])
