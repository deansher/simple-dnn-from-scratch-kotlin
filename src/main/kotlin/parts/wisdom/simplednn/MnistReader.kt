package parts.wisdom.simplednn

import koma.zeros
import koma.extensions.*
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.io.FileInputStream

// Adapted from https://github.com/turkdogan/mnist-data-reader

fun main() {
    val trainingData = readTrainingData()
    printMnistExample(trainingData[trainingData.size - 1])
    val testData = readTestData()
    printMnistExample(testData[0])
}

const val TEST_IMAGE_PATH = "mnist/t10k-images-idx3-ubyte"
const val TEST_LABEL_PATH = "mnist/t10k-labels-idx1-ubyte"
const val TRAIN_IMAGE_PATH = "mnist/train-images-idx3-ubyte"
const val TRAIN_LABEL_PATH = "mnist/train-labels-idx1-ubyte"

fun readTestData(): List<Example> = readData(TEST_IMAGE_PATH, TEST_LABEL_PATH)

fun readTrainingData(): List<Example> = readData(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH)

fun printMnistExample(example: Example) {
    println("label: " + example.label)
    for (r in 0 until example.matrix.numRows()) {
        for (c in 0 until example.matrix.numCols()) {
            print(example.matrix[r, c].toString() + " ")
        }
        println()
    }
}

fun readData(dataFilePath: String, labelFilePath: String): List<Example> {
    val dataInputStream = DataInputStream(
        BufferedInputStream(FileInputStream(dataFilePath))
    )
    val magicNumber = dataInputStream.readInt()
    check(magicNumber == 2051)
    val numberOfItems = dataInputStream.readInt()
    val nRows = dataInputStream.readInt()
    val nCols = dataInputStream.readInt()
    val labelInputStream = DataInputStream(
        BufferedInputStream(FileInputStream(labelFilePath))
    )
    val labelMagicNumber = labelInputStream.readInt()
    check(labelMagicNumber == 2049)
    val numberOfLabels = labelInputStream.readInt()
    val data = mutableListOf<Example>()
    assert(numberOfItems == numberOfLabels)
    repeat(numberOfItems) {
        val label = labelInputStream.readUnsignedByte()
        val m = zeros(nRows, nCols)
        for (r in 0 until nRows) {
            for (c in 0 until nCols) {
                m[r, c] = dataInputStream.readUnsignedByte().toDouble() / 255.0
            }
        }
        data += Example(Coords(0, label), m)
    }
    dataInputStream.close()
    labelInputStream.close()
    return data
}

