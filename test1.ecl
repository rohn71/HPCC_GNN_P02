IMPORT PYTHON3 AS PYTHON;
IMPORT Std;
IMPORT GNN.Tensor;
//IMPORT ImageConverter;
IMPORT tf08;
IMPORT $.Types;



rawImageRec := RECORD
    STRING filename;
    DATA image;
  END;

tensdata := Tensor.R4.tensdata;

ds := DATASET('poke100.flat', rawImageRec, THOR);
OUTPUT(ds);
ds2 := PROJECT(ds,TRANSFORM(Types.ImgRec,self.id:=COUNTER,SELF:=LEFT));

OUTPUT(ds2);

dat := tf08.convertImages(ds2[..5],185,180,1,2);
//dat2 := PROJECT(dat,{UNSIGNED wi, UNSIGNED sliceId, SET OF UNSIGNED shape, UNSIGNED sliceSize, UNSIGNED maxSliceSize});
//dat3 := PROJECT(dat,{SET OF UNSIGNED densedata});


// OUTPUT(dat2[..2],ALL);
// OUTPUT(dat[..5],ALL);
OUTPUT(dat);

//OUTPUT(dat,{nodeId,wi,sliceId,shape,sliceSize});
//OUTPUT(dat);