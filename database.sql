-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Hôte : 127.0.0.1
-- Généré le : jeu. 12 sep. 2024 à 00:17
-- Version du serveur : 10.4.32-MariaDB
-- Version de PHP : 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de données : `ng_django`
--

-- --------------------------------------------------------

--
-- Structure de la table `crud_container`
--

CREATE TABLE `crud_container` (
  `id` bigint(20) NOT NULL,
  `code` varchar(255) NOT NULL,
  `date_time` datetime(6) NOT NULL,
  `detection_threshold` double NOT NULL,
  `image_input` varchar(100) NOT NULL,
  `image_output` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Déchargement des données de la table `crud_container`
--

INSERT INTO `crud_container` (`id`, `code`, `date_time`, `detection_threshold`, `image_input`, `image_output`) VALUES
(29, 'HLXU1395102', '2024-08-25 21:15:11.000000', 1, 'uploads/annotated_image_bd74.jpg', 'uploads/cropped_image_bd74_0.jpg'),
(128, 'CAXU2514229', '2024-08-31 12:49:38.000000', 0.95, 'uploads/annotated_image_6914.jpg', 'uploads/cropped_image_6914_0.jpg'),
(132, 'CPIU185436', '2024-09-04 22:37:46.000000', 0.82, 'uploads/annotated_image_bdee.jpg', 'uploads/cropped_image_bdee_0.jpg');

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `crud_container`
--
ALTER TABLE `crud_container`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT pour les tables déchargées
--

--
-- AUTO_INCREMENT pour la table `crud_container`
--
ALTER TABLE `crud_container`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=133;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
