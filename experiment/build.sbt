val dl4j = "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0"

lazy val commonSettings = Seq(
  organization := "dk.ku",
  version := "0.1.0",
  scalaVersion := "2.12.0"
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "thesis",
    libraryDependencies += dl4j
  )
