import sys
from BitVector import *
from PrimeGenerator import *
from solve_pRoot import *

# Homework Number: 6
# Name: Christopher Chan
# ECN Login: chan328   
# Due Date: 2/27/24

class breakRSA():
    def __init__(self , e) -> None :
        self.e = e
        self.n = None
        self.d = None
        self.p = None
        self.q = None

    # You are free to have other RSA class methods you deem necessary for your solution

    def gcd(self, x, y):
        if y == 0:
            return x
        
        return self.gcd(y, x % y)

    def rsaKeyGen(self):
        generate = PrimeGenerator(bits = 128)
        p = 0
        q = 0

        while(p == q and self.gcd(p, self.e) != 1 and self.gcd(q, self.e) != 1):
            p = generate.findPrime()
            q = generate.findPrime()

        self.p = int(p)
        self.q = int(q)

        #pWrite = open(pFile, "w")
        #qWrite = open(qFile, "w")

        #pWrite.write(str(p))
        #qWrite.write(str(q))

        #pWrite.close()
        #qWrite.close()

    def encrypt ( self , plaintext :str , ciphertext :str ) -> None :
        # your implemenation goes here

        bv = BitVector(filename = plaintext)
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

    def breakEncrypt(self, plaintext :str, ciphertext1 :str, ciphertext2 :str, ciphertext3 :str, modulus) -> None :

        self.rsaKeyGen()
        n1 = self.p * self.q
        self.encrypt(plaintext, ciphertext1)

        self.rsaKeyGen()
        n2 = self.p * self.q
        self.encrypt(plaintext, ciphertext2)

        self.rsaKeyGen()
        n3 = self.p * self.q      
        self.encrypt(plaintext, ciphertext3)

        FILEOUT = open(modulus, "w")
        FILEOUT.write(str(n1))
        FILEOUT.write("\n")
        FILEOUT.write(str(n2))
        FILEOUT.write("\n")
        FILEOUT.write(str(n3))
        FILEOUT.write("\n")

    def breakCrack(self, ciphertext1 :str, ciphertext2 :str, ciphertext3 :str, modulus, recovered_plaintext :str) -> None :
        FILEIN = open(modulus)
        n1 = FILEIN.readline()
        n1 = int(n1)
        n2 = FILEIN.readline()
        n2 = int(n2)
        n3 = FILEIN.readline()
        n3 = int(n3)

        self.n = n1 * n2 * n3

        #Lecture 11.7, Chinese Remainder Theorem
        M1 = n2 * n3
        M1BV = BitVector(intVal = M1)
        M2 = n1 * n3   
        M2BV = BitVector(intVal = M2)
        M3 = n1 * n2
        M3BV = BitVector(intVal = M3)

        #ci = Ni * (MI(Ni) mod ni)
        c1 = M1 * M1BV.multiplicative_inverse(BitVector(intVal = n1)).int_val()
        c2 = M2 * M2BV.multiplicative_inverse(BitVector(intVal = n2)).int_val()
        c3 = M3 * M3BV.multiplicative_inverse(BitVector(intVal = n3)).int_val()

        cipher1 = open(ciphertext1)
        cipher2 = open(ciphertext2)
        cipher3 = open(ciphertext3)

        bv1 = BitVector(hexstring = cipher1.read())
        bv2 = BitVector(hexstring = cipher2.read())
        bv3 = BitVector(hexstring = cipher3.read())

        output = BitVector(size = 0)

        for i in range(len(bv1) // 256):
            bitvec1 = bv1[256*i:256*(i+1)]
            bitvec2 = bv2[256*i:256*(i+1)]
            bitvec3 = bv3[256*i:256*(i+1)]

            #a = m^3 = sum(ai * ci) mod M
            a = (c1 * bitvec1.int_val() + c2 * bitvec2.int_val() + c3 * bitvec3.int_val()) % self.n
            m = solve_pRoot(self.e, a)

            output += BitVector(intVal = m, size=256)[128:] #remove padding

        FILEOUT = open(recovered_plaintext, "w")
        FILEOUT.write(output.get_text_from_bitvector())
        FILEOUT.close()


if __name__ == "__main__":
    cipher = breakRSA( e = 3 )

    if sys.argv[1] == "-e":
        cipher.breakEncrypt (sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif sys.argv[1] == "-c":
        cipher.breakCrack(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])