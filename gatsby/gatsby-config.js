module.exports = {
  pathPrefix: `/solutions`,
  siteMetadata: {
    title: 'solutions.computational.life',
    subtitle: 'sharing computational life solutions across tools and domains',
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
