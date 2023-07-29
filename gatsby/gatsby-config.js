module.exports = {
  pathPrefix: `/`,
  siteMetadata: {
    title: 'solutions.computational.life',
    subtitle: 'sharing computational life',
    catalog_url: 'https://solutions.computational.life',
    menuLinks:[
      {
         name:'Catalog',
         link:'/catalog'
      },
      {
         name:'About',
         link:'/about'
      },
    ]
  },
  plugins: [{ resolve: `gatsby-theme-album`, options: {} }],
}
