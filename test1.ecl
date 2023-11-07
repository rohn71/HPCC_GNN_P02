//This is a test file, where call to ImageConverter module and Types is declared
IMPORT PYTHON3 AS PYTHON;
IMPORT Std;
IMPORT GNN.Tensor;
IMPORT tf08;
IMPORT $.Types;


// another Image RECORD, whith just the fuilename and image contentes, used for TRANSFORM of ImgRec
rawImageRec := RECORD
    STRING filename;
    DATA image;
  END;

tensdata := Tensor.R4.tensdata;

ds := DATASET('poke100.flat', rawImageRec, THOR);
OUTPUT(ds);

//ds2 transforms ImgRec to rawImageRec RECORD,
//it sets the id to a counter for sequential id,
//SELF:=LEFT takes the input record, it basically sets all other attributes to input record (in this case ImgRec)
ds2 := PROJECT(ds,TRANSFORM(Types.ImgRec,self.id:=COUNTER,SELF:=LEFT));

OUTPUT(ds2);

//calling the convertImage module with target shape of the image as parameter
dat := tf08.convertImages(ds2[..5],185,180,1,2);

OUTPUT(dat);
