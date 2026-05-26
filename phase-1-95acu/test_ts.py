import tenseal as ts
import time

def test_seal():
    print("Creating ctx...")
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    
    print("Serializing...")
    ctx_bytes = ctx.serialize(save_secret_key=True)
    
    print("Deserializing...")
    ctx2 = ts.context_from(ctx_bytes)
    
    print("Encrypting...")
    enc = ts.ckks_vector(ctx2, [1.0, 2.0, 3.0])
    print("Decrypted:", enc.decrypt())
    
test_seal()
