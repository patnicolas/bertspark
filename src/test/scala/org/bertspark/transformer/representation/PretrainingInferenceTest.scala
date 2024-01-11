package org.bertspark.transformer.representation

import ai.djl.ndarray.NDManager
import org.apache.spark.sql.SparkSession
import org.bertspark.config.MlopsConfiguration
import org.bertspark.nlp.trainingset._
import org.bertspark.util.io.SingleS3Dataset
import org.bertspark.RuntimeSystemMonitor
import org.bertspark.config.MlopsConfiguration.DebugLog.logDebug
import org.bertspark.transformer.dataset.FeaturesDataset
import org.bertspark.transformer.representation.PretrainingInference.{buildLabeledPretraining, logger}
import org.bertspark.transformer.representation.PretrainingInferenceTest.{createContextualDocument, runtimeSystemMonitor}
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ListBuffer


private[representation] final class PretrainingInferenceTest extends AnyFlatSpec {

  it should "Succeed generating embedding from identical contextual documents" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val contextualDocument = new ContextualDocument(
      "438842_20_7800120504-134244712",
      Array[String]("3_age","m_gender","cornerstone_cust","no_client","xray_modality","21_pos","71046_cpt","26_mod","x-ray","of","chest","","views"),
      "unit xnum exam description xr chest xnum view cpt code xnum primary physician robert ##m os ca do dob xnum xnum xnum attending physician prince steven md rml rll pna sectionimpression moderate right pleural effusion which has slightly worsened compared to the previous exam tiny left effusion patchy bibasilar airspace disease consistent with pneumonia signer name kevin delk md signed xnum xnum xnum xnum xnum pm est workstation name desktop xnumfxnumdxnummp exam xr chest xnum view history rml rll pna comparison xnum xnum xnum sectionfindings heart size appears stable though imp ##ossible to accurately evaluate there is moderate sized right pleural effusion and patchy bibasilar airspace disease com"
    )

    val contextualDocumentDS = Seq[ContextualDocument](contextualDocument, contextualDocument, contextualDocument).toDS()
    val pretrainingInference = new PretrainingInference
    val ndManager = NDManager.newBaseManager()
    val keyedValues = pretrainingInference.predict(ndManager, contextualDocumentDS)
    val embeddings = keyedValues.map(_._2)
    val isEmbedding12Same = embeddings(0).indices.forall(index => embeddings(0)(index) == embeddings(1)(index))
    val isEmbedding13Same = embeddings(0).indices.forall(index => embeddings(0)(index) == embeddings(2)(index))
    println(s"isEmbedding12Same: $isEmbedding12Same,  isEmbedding13Same: $isEmbedding13Same")
    ndManager.close()
  }


  ignore should "Succeed load pre-trained model from S3" in {
    import org.bertspark.implicits._

    val trainingInference = PretrainingInference()
    assert(trainingInference.isPredictorReady == true)
  }

  ignore should "Succeed instantiating a Pre-training model for inference" in {
    import org.bertspark.implicits._

    val pretrainingInference = PretrainingInference()
    assert(pretrainingInference.isPredictorReady)
  }

  ignore should "Succeed predicting from input in S3 folder" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "mlops/CornerstoneTest/contextDocument/AMA"
    val sampleSize = 5
    val s3Dataset = SingleS3Dataset[ContextualDocument](
      s3Folder,
      (contextualDocument: ContextualDocument) => contextualDocument,
      sampleSize
    )
    val ndManager = NDManager.newBaseManager()
    val bertPretrainingInference = PretrainingInference()
    val keyedValues =  bertPretrainingInference.predict(ndManager, s3Dataset)
    val keyedValuesStr = keyedValues.map{ case (docId, values) => s"$docId: ${values.mkString(" ")}"}
    println(keyedValuesStr.mkString("\n"))
    ndManager.close()
  }

  ignore should "Succeed testing extractDocEmbedding" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val contextualDocumentDS = createContextualDocument.toDS()
    val bertFeaturesDataset: FeaturesDataset[ContextualDocument] = buildLabeledPretraining(contextualDocumentDS)
    val ndManager = NDManager.newBaseManager()
    val pretrainingInference = new PretrainingInference
    val keyValues = pretrainingInference.extractDocEmbedding(ndManager, bertFeaturesDataset)
    println(keyValues.map{ case (k,v) => s"$k:$v"}.mkString(" "))
  }

  ignore should "Succeed load predicting from input in S3 folder" in {
    import org.bertspark.implicits._
    import sparkSession.implicits._

    val s3Folder = "mlops/Cornerstone-mini/contextDocument"
    val sampleSize = 5
    val s3Dataset = SingleS3Dataset[ContextualDocument](
      s3Folder,
      (contextualDocument: ContextualDocument) => contextualDocument,
      sampleSize
    )
    val ndManager = NDManager.newBaseManager()
    val bertPretrainingInference = PretrainingInference()

    val runtimeStats = ListBuffer[String]()
    (0 until 360).foreach(
        _ => {
          runtimeStats.append(runtimeSystemMonitor.allMetrics("Test"))
          val keyedValues = bertPretrainingInference.predict(ndManager, s3Dataset)
          val keyedValuesStr = keyedValues.map { case (docId, values) => s"$docId: ${values.mkString(" ")}" }
          println(keyedValuesStr.head)
        }
    )
    println(s"Run time statistics -----------\n${runtimeStats.mkString("\n")}")
    bertPretrainingInference.close()
    ndManager.close()
  }
}


private[representation] final object PretrainingInferenceTest {

  System.setProperty("collect-memory", "true")
  private val runtimeSystemMonitor = new RuntimeSystemMonitor {}

  private def similarityModel(
    ndManager: NDManager,
    s3Dataset: SingleS3Dataset[ContextualDocument],
    pretrainingInference: PretrainingInference
  )(implicit sparkSession: SparkSession): List[KeyedValues] = {
    import sparkSession.implicits._
    val keyedValues = pretrainingInference.predict(ndManager, s3Dataset)
    keyedValues
  }

  private def createContextualDocument: Seq[ContextualDocument] = {
    val contextualDocument1 = ContextualDocument(
      "1",
      Array[String]("0_age","m_gender","cornerstone_cust","no_client","unknown_modality","23_pos","G9637_cpt","no_mod"),
      "dob history number exam ct head without iv contrast indication altered mental status technique helical ct [UNK] the head was obtained without intravenous contrast axial coronal and sagittal images were created dose reduction technique was used including one or more [UNK] the following automated exposure control adjustment [UNK] ma and kv according to patient size and or iterative reconstruction comparison none findings quality average no unusual artifacts beyond the limitations [UNK] routine ct intracranial hemorrhage none mass effect none brain parenchyma normal attenuation characteristics ventricles unremarkable cavum septum pellucidum other none cranial extracranial bony structures no depressed or displaced skull fractures age indeterminate anterior right maxillary wall fracture paranasal sinuses and mastoid air cells the imaged portions appear clear visualized orbits the imaged portions appear normal other right cheek soft tissue swelling parietal scalp contusion impression no acute intracranial findings right cheek soft tissue swelling and age indeterminate anterior right maxillary wall fracture left parietal scalp contusion have personally reviewed the image and the resident interpretation and agree with the findings plat [UNK] by coleman md robert resident curry md res matthew final report"
    )
    val contextualDocument2 = ContextualDocument(
      "2",
      Array[String]("2_age","f_gender","cornerstone_cust","no_client","xray_modality","23_pos","71045_cpt","26_mod"),
      "unit exam description xr chest view portable cpt code primary physician john scott litton md dob attending physician chest pain impression no acute cardiopulmonary findings signer name scott embry md signed am est workstation name finao exam single view [UNK] the chest indication chest pain comparison chest radiograph from findings the cardiomediastinal silhouette is within normal limits no focal consolidation large pleural effusion or evidence [UNK] pneumothorax no acute osseous abnormality"
    )
    val contextualDocument3 = ContextualDocument(
      "3",
      Array[String]("2_age","m_gender","cornerstone_cust","no_client","xray_modality","23_pos","72170_cpt","26_mod"),
      "unit exam description xr pelvis or views cpt code primary physician kandi reece fnp dob attending physician mvc impression no acute process or significant change signer name douglas kaffenberger md signed pm est workstation name finaodk pc exam xr pelvis or views history mvc technique one view comparison findings bone density is normal there is no acute fracture visualized si joints and sacral foramina are unremarkable bilateral hip areas are unremarkable soft tissues are unremarkable"
    )
    val contextualDocument4 = ContextualDocument(
      "4",
      Array[String]("1_age","m_gender","cornerstone_cust","no_client","xray_modality","21_pos","71045_cpt","77_mod"),
      "dob history number examination chest radiograph single view ap or pa clinical history ct removed comparison result lines tubes and devices interval removal left sided chest tube lungs and pleura stable small bilateral pleural effusions with bibasilar atelectasis and or consolidation new small left apical pneumothorax cardiomediastinal silhouette stable size and contour other stable subcutaneous emphysema left chest wall impression new small left apical pneumothorax stable small bilateral pleural effusions with bibasilar atelectasis and or consolidation plat [UNK] by clark md phillip final report"
    )

    Seq[ContextualDocument](contextualDocument1, contextualDocument2, contextualDocument3, contextualDocument4)
  }
}
