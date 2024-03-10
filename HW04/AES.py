import sys
import copy
from BitVector import *

# Homework Number: 4
# Name: Christopher Chan
# ECN Login: chan328   
# Due Date: 2/13/24

class AES ():
# class constructor - when creating an AES object , the
# class ’s constructor is executed and instance variables
# are initialized
    
    def __init__ ( self , keyfile :str ) -> None :
        self.keySize = 256
        FILEIN = open(keyfile)                                                 
        self.givenKey = FILEIN.read()
        self.givenKey = self.givenKey.strip()
        self.givenKey += '0' * (self.keySize//8 - len(self.givenKey)) if len(self.givenKey) < self.keySize//8 else self.givenKey[:self.keySize//8]
        self.key_bv = BitVector( textstring = self.givenKey )
        
        self.AES_modulus = BitVector(bitstring='100011011')
        self.subBytesTable = []                                                  # for encryption
        self.invSubBytesTable = []                                               # for decryption

        self.roundKeys = None

        self.mat = [[0x02, 0x03, 0x01, 0x01],
                    [0x01, 0x02, 0x03, 0x01],
                    [0x01, 0x01, 0x02, 0x03],
                    [0x03, 0x01, 0x01, 0x02]]
        
        self.invmat = [[0x0E, 0x0B, 0x0D, 0x09],
                       [0x09, 0x0E, 0x0B, 0x0D],
                       [0x0D, 0x09, 0x0E, 0x0B],
                       [0x0B, 0x0D, 0x09, 0x0E]]

    def genTables(self):
        c = BitVector(bitstring='01100011')
        d = BitVector(bitstring='00000101')

        for i in range(0, 256):
            # For the encryption SBox
            a = BitVector(intVal = i, size=8).gf_MI(self.AES_modulus, 8) if i != 0 else BitVector(intVal=0)
            # For bit scrambling for the encryption SBox entries:
            a1,a2,a3,a4 = [a.deep_copy() for x in range(4)]
            a ^= (a1 >> 4) ^ (a2 >> 5) ^ (a3 >> 6) ^ (a4 >> 7) ^ c
            self.subBytesTable.append(int(a))

            # For the decryption Sbox:
            b = BitVector(intVal = i, size=8)
            # For bit scrambling for the decryption SBox entries:
            b1,b2,b3 = [b.deep_copy() for x in range(3)]
            b = (b1 >> 2) ^ (b2 >> 5) ^ (b3 >> 7) ^ d
            check = b.gf_MI(self.AES_modulus, 8)
            b = check if isinstance(check, BitVector) else 0
            self.invSubBytesTable.append(int(b))

    def gee(self, keyword, round_constant, byte_sub_table):
        '''
        This is the g() function you see in Figure 4 of Lecture 8.
        '''
        rotated_word = keyword.deep_copy()
        rotated_word << 8
        newword = BitVector(size = 0)
        for i in range(4):
            newword += BitVector(intVal = byte_sub_table[rotated_word[8*i:8*i+8].intValue()], size = 8)
        newword[:8] ^= round_constant
        round_constant = round_constant.gf_multiply_modular(BitVector(intVal = 0x02), self.AES_modulus, 8)
        return newword, round_constant
    
    def gen_subbytes_table(self):
        subBytesTable = []
        c = BitVector(bitstring='01100011')
        for i in range(0, 256):
            a = BitVector(intVal = i, size=8).gf_MI(self.AES_modulus, 8) if i != 0 else BitVector(intVal=0)
            a1,a2,a3,a4 = [a.deep_copy() for x in range(4)]
            a ^= (a1 >> 4) ^ (a2 >> 5) ^ (a3 >> 6) ^ (a4 >> 7) ^ c
            subBytesTable.append(int(a))

        return subBytesTable

    def gen_key_schedule_256(self, key_bv):
        byte_sub_table = self.gen_subbytes_table()
        #  We need 60 keywords (each keyword consists of 32 bits) in the key schedule for
        #  256 bit AES. The 256-bit AES uses the first four keywords to xor the input
        #  block with.  Subsequently, each of the 14 rounds uses 4 keywords from the key
        #  schedule. We will store all 60 keywords in the following list:
        key_words = [None for i in range(60)]
        round_constant = BitVector(intVal = 0x01, size=8)
        for i in range(8):
            key_words[i] = key_bv[i*32 : i*32 + 32]
        for i in range(8,60):
            if i%8 == 0:
                kwd, round_constant = self.gee(key_words[i-1], round_constant, byte_sub_table)
                key_words[i] = key_words[i-8] ^ kwd
            elif (i - (i//8)*8) < 4:
                key_words[i] = key_words[i-8] ^ key_words[i-1]
            elif (i - (i//8)*8) == 4:
                key_words[i] = BitVector(size = 0)
                for j in range(4):
                    key_words[i] += BitVector(intVal = byte_sub_table[key_words[i-1][8*j:8*j+8].intValue()], size = 8)
                key_words[i] ^= key_words[i-8] 
            elif ((i - (i//8)*8) > 4) and ((i - (i//8)*8) < 8):
                key_words[i] = key_words[i-8] ^ key_words[i-1]
            else:
                sys.exit("error in key scheduling algo for i = %d" % i)
        return key_words

    def genRoundKeys(self):
        key_words = []
        keysize = self.keySize
        key_bv = self.key_bv

        if keysize == 256:    
            key_words = self.gen_key_schedule_256(key_bv)
        else:
            sys.exit("wrong keysize --- aborting")

        key_schedule = []

        for wordIndex,word in enumerate(key_words):
            #print(type(word))
            keyword_in_ints = []
            for i in range(4):
                keyword_in_ints.append(word[i*8:i*8+8].intValue()) 
            #if word_index % 4 == 0: print("\n")
            #print("word %d:  %s" % (word_index, str(keyword_in_ints)))
            key_schedule.append(keyword_in_ints)

        num_rounds = None
        if keysize == 256: num_rounds = 14
        round_keys = [None for i in range(num_rounds + 1)]

        for i in range(num_rounds + 1):
            round_keys[i] = (key_words[i*4] + key_words[i*4+1] + key_words[i*4+2] + key_words[i*4+3])

        self.roundKeys = round_keys

    def subBytes(self, state):
        for i in range(4):
            for j in range(4):
                state[i][j] = BitVector(intVal = self.subBytesTable[state[i][j].intValue()], size=8)

        return state
    
    def inverseSubBytes(self, state):
        for i in range(4):
            for j in range(4):
                state[j][i] = BitVector(intVal = self.invSubBytesTable[state[j][i].intValue()], size=8)

        return state

    def shiftRows(self, state):
        for i in range(1, 4):
            state[i] = state[i][i:] + state[i][:i]

        return state
    
    def inverseShiftRows(self, state):
        for i in range(1,4):
            state[i] = state[i][-i:] + state[i][:-i]

        return state

    def mixColumns(self, state):
        
        result = BitVector(size = 0)
        for j in range(4):
            column = [state[i][j] for i in range(4)]
            for i in range(4):
                value = BitVector(intVal = 0, size = 8)
                for k in range(4):
                    multiplier = BitVector(intVal = self.mat[i][k], size = 8)
                    value ^= multiplier.gf_multiply_modular(column[k], self.AES_modulus, 8)
                    
                result += value

        state = self.bVToMatrix(result)

        '''
        stateCpy = copy.deepcopy(state) #copy the entire list of bvs
        factor = [BitVector(hexstring="02"), BitVector(hexstring="03"), BitVector(hexstring="01"), BitVector(hexstring="01")]
        for j in range(4):
            for i in range(4):
                temp = BitVector(size=8)

                for k in range(4):
                    temp ^= stateCpy[k][j].gf_multiply_modular(factor[k - i], self.AES_modulus, 8)

                state[i][j] = temp
        '''
        return state
    
    def inverseMixColumns(self, state):
        '''
        stateCpy = copy.deepcopy(state)
        factor = [BitVector(hexstring="0E"), BitVector(hexstring="0B"), BitVector(hexstring="0D"), BitVector(hexstring="09")]
        for j in range(4):
            for i in range(4):
                temp = BitVector(size=8)

                for k in range(4):  
                    temp ^= stateCpy[k][j].gf_multiply_modular(factor[k - i], self.AES_modulus, 8)

                state[i][j] = temp
        '''
        
        result = BitVector(size = 0)
        for j in range(4):
            column = [state[i][j] for i in range(4)]
            for i in range(4):
                value = BitVector(intVal = 0, size = 8)
                for k in range(4):
                    multiplier = BitVector(intVal = self.invmat[i][k], size = 8)
                    value ^= multiplier.gf_multiply_modular(column[k], self.AES_modulus, 8)

                result += value

        state = self.bVToMatrix(result)

        return state

    def bVToMatrix(self, bv):
        return [[bv[j*32+i*8:j*32+i*8+8] for j in range(4)] for i in range(4)]
    
    def matrixToBV(self, state):
        combined = BitVector(size = 0)
        for i in range(4):
                for j in range(4):
                    combined = combined + state[j][i]

        return combined

    # encrypt - method performs AES encryption on the plaintext and writes the ciphertext to disk
    # Inputs : plaintext (str) - filename containing plaintext
    # ciphertext (str) - filename containing ciphertext
    # Return : void
    def encrypt ( self , plaintext :str , ciphertext :str ) -> None :
        self.genTables()
        self.genRoundKeys()
        bv = BitVector(filename = plaintext)
        encryptedText = ''
        state = [[0] * 4] * 4

        while (bv.more_to_read):
            bitvec = bv.read_bits_from_file( 128 )

            #pad with 0s if less than 128 bits
            if bitvec.length() < 128:
                bitvec.pad_from_right(128 - bitvec.length())
            
            #print("step 1:" + bitvec.get_bitvector_in_hex())

            roundNum = 0
            bitvec ^= self.roundKeys[roundNum]
            #print('step 2:' + bitvec.get_bitvector_in_hex())

            #rounds 2-13
            for j in range(1, 14):
                state = self.bVToMatrix(bitvec)
                state = self.subBytes(state)
                #combined = self.matrixToBV(state)
                #print("step 3:" + combined.get_bitvector_in_hex())

                state = self.shiftRows(state)
                #combined = self.matrixToBV(state)
                #print("step 4:" + combined.get_bitvector_in_hex())

                state = self.mixColumns(state)
                #combined = self.matrixToBV(state)
                #print("step 5:" + combined.get_bitvector_in_hex())

                roundNum = j
                bitvec = self.matrixToBV(state)
                bitvec ^= self.roundKeys[roundNum]
                #print("step 6:" + bitvec.get_bitvector_in_hex())
            
            #final round
            state = self.bVToMatrix(bitvec)
            state = self.subBytes(state)
            state = self.shiftRows(state)
            roundNum += 1
            bitvec = self.matrixToBV(state)
            bitvec ^= self.roundKeys[roundNum] 

            encryptedText = encryptedText + bitvec.get_hex_string_from_bitvector()
            #print("block" + encryptedText) #doesnt print for some reason :|
            

        #encryptedText = encryptedText.get_hex_string_from_bitvector()
        #print("final:" + encryptedText) #doesnt print for some reason :|

        # Write ciphertext bitvector to the output file:
        FILEOUT = open(ciphertext, 'w')                                              
        FILEOUT.write(encryptedText)                                                      
        FILEOUT.close()

    # decrypt - method performs AES decryption on the ciphertext and writes the
    # recovered plaintext to disk
    # Inputs : ciphertext (str) - filename containing ciphertext
    # decrypted (str) - filename containing recovered plaintext
    # Return : void
    def decrypt ( self , ciphertext :str , decrypted :str ) -> None :
        self.genTables()
        self.genRoundKeys()
        FILEIN = open(ciphertext)
        bv = BitVector( hexstring = FILEIN.read() )
        FILEOUT = open(decrypted, 'w', encoding="utf-8")
        decryptedText = BitVector(size = 0)
        state = [[0] * 4] * 4

        for i in range(len(bv) // 128):
            bitvec = bv[128*i:128*(i+1)]

            #pad with 0s if less than 128 bits
            if bitvec.length() < 128:
                bitvec.pad_from_right(128 - bitvec.length())
            
            #print("step 1:" + bitvec.get_bitvector_in_hex())

            #roundNum = 13
            bitvec ^= self.roundKeys[-1] # 0
            #print('step 2:' + bitvec.get_bitvector_in_hex())

            #rounds 2-14
            for j in range(1, 14): #
                state = self.bVToMatrix(bitvec)
                state = self.inverseShiftRows(state)
                combined = self.matrixToBV(state)
                #print("step 3:" + combined.get_bitvector_in_hex())
                
                state = self.inverseSubBytes(state)
                combined = self.matrixToBV(state)
                #print("step 4:" + combined.get_bitvector_in_hex())

                #roundNum = 13 - j
                bitvec = self.matrixToBV(state)
                bitvec ^= self.roundKeys[14 - j]
                combined = self.matrixToBV(state)
                #print("step 5:" + combined.get_bitvector_in_hex())
                
                state = self.bVToMatrix(bitvec)
                state = self.inverseMixColumns(state)
                bitvec = self.matrixToBV(state)
                #print("step 6:" + bitvec.get_bitvector_in_hex())

                #final round case
                if j == 13:
                    state = self.bVToMatrix(bitvec)
                    state = self.inverseShiftRows(state)
                    state = self.inverseSubBytes(state)
                    #roundNum = 0
                    bitvec = self.matrixToBV(state)
                    bitvec ^= self.roundKeys[0]
                    break
            
            '''
            #final round
            state = self.bVToMatrix(bitvec)
            state = self.inverseShiftRows(state)
            state = self.inverseSubBytes(state)
            roundNum = 0
            bitvec = self.matrixToBV(state)
            bitvec ^= self.roundKeys[roundNum]
            '''
            
            decryptedText = decryptedText + bitvec
            FILEOUT.write(bitvec.get_bitvector_in_ascii())
            #bitvec.write_to_file(FILEOUT) 
            #print("block: " + bitvec.get_hex_string_from_bitvector())
            
        #print("final: " + decryptedText.get_bitvector_in_ascii())

        # Write ciphertext bitvector to the output file:
        
        #FILEOUT.write(decryptedText.get_bitvector_in_ascii())                                                                                                          
        FILEOUT.close()

if __name__ == "__main__":
    cipher = AES ( keyfile = sys . argv [3])

    if sys . argv [1] == "-e":
        cipher.encrypt( plaintext = sys.argv [2], ciphertext = sys.argv [4])
    elif sys . argv [1] == "-d":
        cipher.decrypt( ciphertext = sys.argv [2], decrypted = sys.argv [4])
    else :
        sys.exit(" Incorrect Command - Line Syntax ")