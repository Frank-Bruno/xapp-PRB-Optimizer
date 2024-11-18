from defs import E2SM_KPM_RC
from defs import E2AP_IEs, E2AP_PDU_Contents, E2AP_PDU_Descriptions

# RAN Function Definition
encoded = "34164F52414E2D5747332D4B504D02804F494431323305004B504D206D6F6E69746F7201000001010700506572696F646963207265706F727401010001011E804F2D43552D4350204D6561737572656D656E7420436F6E7461696E657220666F72207468652045504320636F6E6E6563746564206465706C6F796D656E7401010101"

# Convert the encoded hex string to bytes
encoded_bytes = bytes.fromhex(encoded)
print(encoded_bytes)


try:
    print(E2SM_KPM_RC.E2SM_RC_RANFunctionDefinition.from_aper(encoded_bytes))
except Exception as e:
    print(f"Error decoding E2SM_RC_RANFunctionDefinition: {e}")

try:
    print(E2SM_KPM_RC.E2SM_KPM_RANFunctionDefinition.from_aper(encoded_bytes))
except Exception as e:
    print(f"Error decoding E2SM_KPM_RANFunctionDefinition: {e}")

try:
    print(E2AP_IEs.RANfunctionDefinition.from_aper(encoded_bytes))
except Exception as e:
    print(f"Error decoding RANfunctionDefinition: {e}")

try:
    print(E2AP_PDU_Contents.RANfunction_Item.from_aper(encoded_bytes))
except Exception as e:
    print(f"Error decoding RANfunction_Item: {e}")