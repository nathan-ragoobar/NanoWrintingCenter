#include "mainwindow.h"
#include <QProcess>
#include <QDir>
#include <QCoreApplication>
#include <QTemporaryFile>

bool InferenceInterface::modelLoaded = false;

bool InferenceInterface::loadModel() {
    QProcess process;
    QString program = QDir(QCoreApplication::applicationDirPath()).filePath("inference_cli");
    QStringList arguments = {"--check-model"};
    
    process.start(program, arguments);
    if (!process.waitForFinished(30000)) {
        return false;
    }
    
    bool success = (process.exitCode() == 0);
    modelLoaded = success;
    return success;
}

QString InferenceInterface::generateResponse(const QString& input, int maxTokens) {
    QTemporaryFile inputFile;
    if (!inputFile.open()) {
        return "Error: Failed to create temporary input file";
    }
    
    inputFile.write(input.toUtf8());
    inputFile.flush();
    
    QTemporaryFile outputFile;
    if (!outputFile.open()) {
        return "Error: Failed to create temporary output file";
    }
    
    QProcess process;
    QString program = QDir(QCoreApplication::applicationDirPath()).filePath("inference_cli");
    QStringList arguments;
    arguments << inputFile.fileName()
             << QString::number(maxTokens)
             << outputFile.fileName();
    
    process.start(program, arguments);
    
    if (!process.waitForFinished(60000)) { // 60-second timeout
        return "Error: Inference process timed out";
    }
    
    if (process.exitCode() != 0) {
        QString error = QString::fromUtf8(process.readAllStandardError());
        return "Error: " + error;
    }
    
    outputFile.seek(0);
    QString response = QString::fromUtf8(outputFile.readAll());
    return response;
}